import Foundation
import Metal

/// Qwen Sparse Attention's block indexer.
///
/// A small head set (4 heads of 128 against the attention's 24 of 256) scores
/// mean-pooled blocks of keys, and the attention is then restricted to the
/// highest-scoring blocks plus the ragged tail of the current query's own
/// block. Below `QSAExactness.maximumExactVisibleKeys` the selection is
/// vacuous and dense attention computes the same thing; past it, this is what
/// makes the model faithful rather than merely fluent.
///
/// Two caches, both per QSA layer:
///
/// - the **raw** indexer keys, cached before their norm and rope exactly as
///   the reference does, because a block's pooled vector is the mean of raw
///   members and normalizing first would change what is averaged;
/// - the **pooled** block vectors, post-norm and post-rope. Only the block
///   holding the newest token changes on a decode step, so a step repools one
///   block and leaves the history alone.
///
/// The block's rope position is `blockIndex * compressRatio` -- the position
/// of its first member, not the query's.
final class QSAIndexer {
    struct Weights {
        let queryProjection: TensorView
        let keyProjection: TensorView
        let queryNorm: TensorView
        let keyNorm: TensorView
    }

    let heads: Int
    let headDim: Int
    let compressRatio: Int
    let budget: Int
    let ropeTheta: Float

    private let ctx: MetalContext
    private let poolPSO: MTLComputePipelineState
    private let poolRangePSO: MTLComputePipelineState
    private let scorePSO: MTLComputePipelineState
    private let scoreRowsPSO: MTLComputePipelineState
    private let rms: RMSNorm
    private let rope: RoPE
    private let gemv: SlotGEMV

    /// `[heads * headDim]` query scratch, normed and roped in place.
    private let queryBuf: MTLBuffer
    /// One token's raw key, before it joins the cache.
    private let keyBuf: MTLBuffer
    /// Per layer: `[capacity, headDim]` raw keys and
    /// `[capacity / compressRatio + 1, headDim]` pooled blocks.
    private var rawKeys: [Int: MTLBuffer] = [:]
    private var pooled: [Int: MTLBuffer] = [:]
    /// `[n_blocks]` float scores, and the `[capacity]` byte selection.
    private var scoresBuf: MTLBuffer
    private var keepBuf: MTLBuffer
    /// Compacted selection: `selectionWidth` ascending key indices per query,
    /// plus how many of them are valid. The mask form costs the attention
    /// kernel one byte-load per *visible* key per query per head, which is
    /// O(context) work to discover an O(budget) answer -- measured at 4.53s per
    /// layer-chunk at 10k context against 1.42s at 2.4k, where context and
    /// budget coincide. Compacted, the loop is the budget regardless of depth.
    private var keepIndexBuf: MTLBuffer?
    private var keepCountBuf: MTLBuffer?
    private(set) var capacity: Int
    /// Chunk-sized scratch, grown on demand: a prefill chunk needs a query
    /// row and a selection row per token, which decode does not.
    private var queryRowsBuf: MTLBuffer?
    private var lastScoredBlocks = 0
    /// Blocks scored per query row by the most recent chunk pass.
    var scoredBlocksPerRow: Int { lastScoredBlocks }

    /// Cells a query keeps: the budget plus the tail block's extra members.
    var selectionWidth: Int { budget + compressRatio - 1 }

    init(context: MetalContext, config: SparseIndexerConfig,
         budget: Int? = nil,
         ropeTheta: Float, capacity: Int, weightBits: Int = 4) throws {
        precondition(config.enabled, "QSAIndexer requires a configured indexer")
        precondition(capacity > 0)
        self.ctx = context
        self.heads = config.numHeads
        self.headDim = config.headDim
        self.compressRatio = config.compressRatio
        self.budget = budget ?? config.budget
        self.ropeTheta = ropeTheta
        self.capacity = capacity
        self.poolPSO = try context.pipeline("qsa_pool_block")
        self.poolRangePSO = try context.pipeline("qsa_pool_blocks")
        self.scorePSO = try context.pipeline("qsa_block_scores")
        self.scoreRowsPSO = try context.pipeline("qsa_block_scores_rows")
        self.rms = try RMSNorm(context: context)
        self.rope = try RoPE(context: context)
        self.gemv = try SlotGEMV(context: context, weightBits: weightBits)
        let f16 = MemoryLayout<Float16>.stride
        guard let query = context.device.makeBuffer(
                  length: heads * headDim * f16, options: .storageModeShared),
              let key = context.device.makeBuffer(
                  length: headDim * f16, options: .storageModeShared),
              let scores = context.device.makeBuffer(
                  length: max(1, capacity / compressRatio + 1)
                      * MemoryLayout<Float>.stride,
                  options: .storageModeShared),
              let keep = context.device.makeBuffer(
                  length: capacity, options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed("QSA indexer scratch")
        }
        self.queryBuf = query
        self.keyBuf = key
        self.scoresBuf = scores
        self.keepBuf = keep
    }

    /// The most recent selection and the scores behind it, for A/B-ing two
    /// paths that should agree.
    /// The pooled block cache for one layer, post-norm and post-rope.
    func debugPooled(layer: Int, blocks: Int) -> [Float]? {
        guard let pool = pooled[layer] else { return nil }
        let ptr = pool.contents().bindMemory(to: Float16.self,
                                             capacity: blocks * headDim)
        return (0..<blocks * headDim).map { Float(ptr[$0]) }
    }

    /// The raw indexer keys for one layer, before norm and rope.
    func debugRawKeys(layer: Int, count: Int) -> [Float]? {
        guard let raw = rawKeys[layer] else { return nil }
        let ptr = raw.contents().bindMemory(to: Float16.self,
                                            capacity: count * headDim)
        return (0..<count * headDim).map { Float(ptr[$0]) }
    }

    func debugSnapshot(cells: Int, blocks: Int) -> (keep: [UInt8], scores: [Float]) {
        let k = keepBuf.contents().bindMemory(to: UInt8.self, capacity: cells)
        let sc = scoresBuf.contents().bindMemory(to: Float.self, capacity: blocks)
        return ((0..<cells).map { k[$0] }, (0..<blocks).map { sc[$0] })
    }

    /// The block-score scratch, so a test can drive the selection with known
    /// rankings instead of whatever the scoring kernel happens to produce.
    var scoresForTesting: MTLBuffer { scoresBuf }

    /// The selection buffer the attention reads, or nil when every visible
    /// key is kept and the mask would be a no-op.
    var keepMask: MTLBuffer { keepBuf }

    func reset() {
        rawKeys.removeAll()
        pooled.removeAll()
    }

    private func layerBuffers(_ layer: Int) throws -> (raw: MTLBuffer, pooled: MTLBuffer) {
        if let raw = rawKeys[layer], let pool = pooled[layer] {
            return (raw, pool)
        }
        // Sized to the full context on a layer's first use, unlike the KV
        // cache which grows on demand. At the production geometry that is
        // 1.25 MB a layer for a 4,096-token context and about 84 MB at the
        // model's full 262,144 -- so growing on demand is worth doing if long
        // contexts become routine, and is not worth the bookkeeping yet.
        let f16 = MemoryLayout<Float16>.stride
        guard let raw = ctx.device.makeBuffer(
                  length: capacity * headDim * f16, options: .storageModeShared),
              let pool = ctx.device.makeBuffer(
                  length: (capacity / compressRatio + 1) * headDim * f16,
                  options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed("QSA indexer layer \(layer)")
        }
        raw.label = "qsa.rawKeys.L\(layer)"
        pool.label = "qsa.pooled.L\(layer)"
        rawKeys[layer] = raw
        pooled[layer] = pool
        return (raw, pool)
    }

    /// Appends this token's raw indexer key and repools the block it lands
    /// in. `position` is the token's absolute position.
    func encodeAppendKey(commandBuffer: MTLCommandBuffer,
                         hidden: MTLBuffer, hiddenOffset: Int = 0,
                         weights: Weights, layer: Int, position: Int,
                         eps: Float) throws {
        precondition(position < capacity,
                     "QSA indexer capacity \(capacity) exceeded at \(position)")
        let buffers = try layerBuffers(layer)
        let key = weights.keyProjection
        try gemv.encode(commandBuffer: commandBuffer,
                        weights: key.buffer, weightsOffset: Int(key.offset),
                        scales: key.buffer, scalesOffset: Int(key.scaleOffset),
                        biases: key.buffer, biasesOffset: Int(key.biasOffset),
                        x: hidden, xOffset: hiddenOffset,
                        y: buffers.raw,
                        yOffset: position * headDim * MemoryLayout<Float16>.stride,
                        m: UInt32(headDim), n: UInt32(hiddenColumns(key)),
                        isBF16: key.dtype == 1)

        // Repool the block this token joined. Members present is what the
        // mean divides by, so a tail block is not diluted by absent members.
        let block = position / compressRatio
        let first = block * compressRatio
        let count = position - first + 1
        try encodePool(commandBuffer: commandBuffer, buffers: buffers,
                       block: block, first: first, count: count)
        try rms.encodeBF16W(commandBuffer: commandBuffer,
                            x: buffers.pooled,
                            xOffset: block * headDim * MemoryLayout<Float16>.stride,
                            weight: weights.keyNorm.buffer,
                            weightOffset: Int(weights.keyNorm.offset),
                            out: buffers.pooled,
                            outOffset: block * headDim * MemoryLayout<Float16>.stride,
                            d: UInt32(headDim), eps: eps)
        try rope.encodeNeoxSubdim(commandBuffer: commandBuffer,
                              data: buffers.pooled,
                              dataOffset: block * headDim * MemoryLayout<Float16>.stride,
                              position: UInt32(first),
                              headDim: UInt32(headDim), numHeads: UInt32(1),
                              rotaryDim: UInt32(headDim), theta: ropeTheta)
    }

    /// Columns of a `[rows, columns]` projection, from the recorded shape.
    private func hiddenColumns(_ view: TensorView) -> Int {
        Int(view.shape.1)
    }

    private func encodePool(commandBuffer: MTLCommandBuffer,
                            buffers: (raw: MTLBuffer, pooled: MTLBuffer),
                            block: Int, first: Int, count: Int) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(poolPSO)
        enc.setBuffer(buffers.raw, offset: 0, index: 0)
        enc.setBuffer(buffers.pooled, offset: 0, index: 1)
        var d = UInt32(headDim), b = UInt32(block)
        var f = UInt32(first), c = UInt32(count)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&f, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 5)
        let w = min(poolPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: headDim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The first block of a chunk may already hold members from the previous
    /// one, and the last may be ragged; repooling from
    /// `startPosition / compressRatio` through the end covers both without a
    /// special case for either.
    /// Repools every block a prefill chunk touched. The caller has already
    /// projected the chunk's raw keys into the cache at
    /// `rawKeyDestination(layer:startPosition:)`.
    func encodePoolPrefill(commandBuffer: MTLCommandBuffer,
                           weights: Weights, layer: Int,
                           startPosition: Int, tokens: Int,
                           eps: Float) throws {
        precondition(startPosition + tokens <= capacity,
                     "QSA indexer capacity \(capacity) exceeded at "
                         + "\(startPosition + tokens)")
        let buffers = try layerBuffers(layer)
        let f16 = MemoryLayout<Float16>.stride
        let endPosition = startPosition + tokens
        let firstBlock = startPosition / compressRatio
        let lastBlock = (endPosition - 1) / compressRatio
        let blockCount = lastBlock - firstBlock + 1
        try encodePoolRange(commandBuffer: commandBuffer, buffers: buffers,
                            firstBlock: firstBlock, blockCount: blockCount,
                            visibleKeys: endPosition)
        try rms.encodeBF16WPerHead(commandBuffer: commandBuffer,
                                   x: buffers.pooled,
                                   xOffset: firstBlock * headDim * f16,
                                   weight: weights.keyNorm.buffer,
                                   weightOffset: Int(weights.keyNorm.offset),
                                   out: buffers.pooled,
                                   outOffset: firstBlock * headDim * f16,
                                   headDim: UInt32(headDim),
                                   numHeads: blockCount, eps: eps)
        try rope.encodeNeoxSubdimStrided(
            commandBuffer: commandBuffer,
            data: buffers.pooled,
            dataOffset: firstBlock * headDim * f16,
            position: UInt32(firstBlock * compressRatio),
            headDim: UInt32(headDim), numHeads: 1,
            rotaryDim: UInt32(headDim), numTokens: UInt32(blockCount),
            stride: UInt32(compressRatio), theta: ropeTheta)
    }

    /// The raw-key cache slice a prefill chunk's projection should write into.
    func rawKeyDestination(layer: Int, startPosition: Int) throws
        -> (buffer: MTLBuffer, offset: Int) {
        let buffers = try layerBuffers(layer)
        return (buffers.raw,
                startPosition * headDim * MemoryLayout<Float16>.stride)
    }

    private func encodePoolRange(commandBuffer: MTLCommandBuffer,
                                 buffers: (raw: MTLBuffer, pooled: MTLBuffer),
                                 firstBlock: Int, blockCount: Int,
                                 visibleKeys: Int) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(poolRangePSO)
        enc.setBuffer(buffers.raw, offset: 0, index: 0)
        enc.setBuffer(buffers.pooled, offset: 0, index: 1)
        var d = UInt32(headDim), fb = UInt32(firstBlock)
        var bc = UInt32(blockCount), r = UInt32(compressRatio)
        var n = UInt32(visibleKeys)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&fb, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&bc, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&r, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 6)
        let w = min(poolRangePSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(
            MTLSize(width: headDim * blockCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Scores every block against this token's indexer query.
    func encodeScores(commandBuffer: MTLCommandBuffer,
                      hidden: MTLBuffer, hiddenOffset: Int = 0,
                      weights: Weights, layer: Int, position: Int,
                      eps: Float) throws {
        let buffers = try layerBuffers(layer)
        let query = weights.queryProjection
        try gemv.encode(commandBuffer: commandBuffer,
                        weights: query.buffer, weightsOffset: Int(query.offset),
                        scales: query.buffer, scalesOffset: Int(query.scaleOffset),
                        biases: query.buffer, biasesOffset: Int(query.biasOffset),
                        x: hidden, xOffset: hiddenOffset,
                        y: queryBuf,
                        m: UInt32(heads * headDim), n: UInt32(hiddenColumns(query)),
                        isBF16: query.dtype == 1)
        try rms.encodeBF16WPerHead(commandBuffer: commandBuffer,
                                   x: queryBuf,
                                   weight: weights.queryNorm.buffer,
                                   weightOffset: Int(weights.queryNorm.offset),
                                   out: queryBuf,
                                   headDim: UInt32(headDim), numHeads: heads,
                                   eps: eps)
        try rope.encodeNeoxSubdim(commandBuffer: commandBuffer,
                              data: queryBuf,
                              position: UInt32(position),
                              headDim: UInt32(headDim), numHeads: UInt32(heads),
                              rotaryDim: UInt32(headDim), theta: ropeTheta)

        let blocks = position / compressRatio + 1
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(scorePSO)
        enc.setBuffer(queryBuf, offset: 0, index: 0)
        enc.setBuffer(buffers.pooled, offset: 0, index: 1)
        enc.setBuffer(scoresBuf, offset: 0, index: 2)
        var d = UInt32(headDim), h = UInt32(heads), b = UInt32(blocks)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&h, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 5)
        let w = min(scorePSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: blocks, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Scores every block against every query in a prefill chunk.
    ///
    /// `blockInput` is the chunk's `[tokens, hidden]` block input; the query
    /// projection, its norm and its rope all run over the whole chunk, with
    /// the rope advancing one position per row.
    func encodeScoresPrefill(commandBuffer: MTLCommandBuffer,
                             blockInput: MTLBuffer,
                             weights: Weights, layer: Int,
                             startPosition: Int, tokens: Int,
                             eps: Float,
                             project: (MTLCommandBuffer, TensorView, MTLBuffer,
                                       MTLBuffer, Int, Int, Int) throws -> Void) throws {
        let buffers = try layerBuffers(layer)
        let query = weights.queryProjection
        try growQueryScratch(rows: tokens)
        try project(commandBuffer, query, blockInput, queryRowsBuf!,
                    heads * headDim, hiddenColumns(query), tokens)
        try rms.encodeBF16WPerHead(commandBuffer: commandBuffer,
                                   x: queryRowsBuf!,
                                   weight: weights.queryNorm.buffer,
                                   weightOffset: Int(weights.queryNorm.offset),
                                   out: queryRowsBuf!,
                                   headDim: UInt32(headDim),
                                   numHeads: heads * tokens, eps: eps)
        try rope.encodeNeoxSubdimStrided(
            commandBuffer: commandBuffer, data: queryRowsBuf!,
            position: UInt32(startPosition),
            headDim: UInt32(headDim), numHeads: UInt32(heads),
            rotaryDim: UInt32(headDim), numTokens: UInt32(tokens),
            stride: 1, theta: ropeTheta)

        let blocks = (startPosition + tokens - 1) / compressRatio + 1
        try growScoreScratch(count: blocks * tokens)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.commandEncoderFailed
        }
        enc.setComputePipelineState(scoreRowsPSO)
        enc.setBuffer(queryRowsBuf!, offset: 0, index: 0)
        enc.setBuffer(buffers.pooled, offset: 0, index: 1)
        enc.setBuffer(scoresBuf, offset: 0, index: 2)
        var d = UInt32(headDim), h = UInt32(heads)
        var b = UInt32(blocks), t = UInt32(tokens)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&h, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&b, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 6)
        let w = min(scoreRowsPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: blocks * tokens, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        lastScoredBlocks = blocks
    }

    /// The chunk's selection, or nil when every query in the chunk is inside
    /// the dense-exact window. Carries both forms: the mask that existing
    /// callers and the activation dump read, and the compacted index list the
    /// attention kernel loops over.
    ///
    /// Every query gets the same treatment the decode path gives one: its own
    /// ragged tail forced in, then complete blocks by score until the cell
    /// budget runs out. A query inside the window keeps everything it can see.
    func selectKeysPrefill(startPosition: Int, tokens: Int)
        -> QSASelection? {
        let lastVisible = startPosition + tokens
        guard lastVisible > selectionWidth else { return nil }
        let stride = lastVisible
        guard (try? growKeepScratch(count: stride * tokens)) != nil else { return nil }
        // One index slot per key the budget can admit. The dense-window rows
        // keep `visible` keys, which is at most selectionWidth by the guard
        // above, so this width covers both branches.
        let indexWidth = selectionWidth
        guard (try? growCompactScratch(rows: tokens, width: indexWidth)) != nil,
              let indexBuf = keepIndexBuf, let countBuf = keepCountBuf
        else { return nil }
        let keep = keepBuf.contents().bindMemory(to: UInt8.self,
                                                 capacity: stride * tokens)
        let indices = indexBuf.contents().bindMemory(to: UInt32.self,
                                                     capacity: tokens * indexWidth)
        let counts = countBuf.contents().bindMemory(to: UInt32.self, capacity: tokens)
        let scores = scoresBuf.contents().bindMemory(
            to: Float.self, capacity: lastScoredBlocks * tokens)
        for row in 0..<tokens {
            let visible = startPosition + row + 1
            let base = row * stride
            memset(keep + base, 0, stride)
            if visible <= selectionWidth {
                memset(keep + base, 1, visible)
                let out = indices + row * indexWidth
                for key in 0..<visible { out[key] = UInt32(key) }
                counts[row] = UInt32(visible)
                continue
            }
            let completeCells = (visible / compressRatio) * compressRatio
            for cell in completeCells..<visible { keep[base + cell] = 1 }
            var remaining = selectionWidth - (visible - completeCells)
            guard remaining > 0 else { continue }
            let blocks = completeCells / compressRatio
            let rowScores = scores + row * lastScoredBlocks
            let ranked = (0..<blocks).sorted {
                rowScores[$0] == rowScores[$1]
                    ? $0 < $1 : rowScores[$0] > rowScores[$1]
            }
            for block in ranked {
                if remaining <= 0 { break }
                let take = min(compressRatio, remaining)
                let cellBase = base + block * compressRatio
                for offset in 0..<take { keep[cellBase + offset] = 1 }
                remaining -= take
            }
            // Compact ascending. Scanning the row once here is O(visible) on
            // the host and replaces the same scan done once per query *per
            // head* on the GPU; ascending order also keeps the K/V reads
            // sequential, which a score-ordered list would not.
            var written = 0
            let out = indices + row * indexWidth
            for key in 0..<visible where keep[base + key] != 0 {
                if written == indexWidth { break }
                out[written] = UInt32(key)
                written += 1
            }
            counts[row] = UInt32(written)
        }
        return QSASelection(mask: keepBuf, maskStride: stride,
                            indices: indexBuf, indexStride: indexWidth,
                            counts: countBuf)
    }

    private func growQueryScratch(rows: Int) throws {
        let needed = rows * heads * headDim * MemoryLayout<Float16>.stride
        if let existing = queryRowsBuf, existing.length >= needed { return }
        guard let made = ctx.device.makeBuffer(length: needed,
                                               options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed("QSA indexer chunk queries")
        }
        made.label = "qsa.chunkQueries"
        queryRowsBuf = made
    }

    private func growScoreScratch(count: Int) throws {
        let needed = count * MemoryLayout<Float>.stride
        if scoresBuf.length >= needed { return }
        guard let made = ctx.device.makeBuffer(length: needed,
                                               options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed("QSA indexer chunk scores")
        }
        made.label = "qsa.scores"
        scoresBuf = made
    }

    private func growCompactScratch(rows: Int, width: Int) throws {
        let idxBytes = rows * width * MemoryLayout<UInt32>.stride
        let cntBytes = rows * MemoryLayout<UInt32>.stride
        if keepIndexBuf == nil || keepIndexBuf!.length < idxBytes {
            guard let made = ctx.device.makeBuffer(length: idxBytes,
                                                   options: .storageModeShared) else {
                throw MetalError.bufferAllocationFailed("QSA selection indices")
            }
            made.label = "qsa.keep.indices"
            keepIndexBuf = made
        }
        if keepCountBuf == nil || keepCountBuf!.length < cntBytes {
            guard let made = ctx.device.makeBuffer(length: cntBytes,
                                                   options: .storageModeShared) else {
                throw MetalError.bufferAllocationFailed("QSA selection counts")
            }
            made.label = "qsa.keep.counts"
            keepCountBuf = made
        }
    }

    private func growKeepScratch(count: Int) throws {
        if keepBuf.length >= count { return }
        guard let made = ctx.device.makeBuffer(length: count,
                                               options: .storageModeShared) else {
            throw MetalError.bufferAllocationFailed("QSA indexer chunk selection")
        }
        made.label = "qsa.keep"
        keepBuf = made
    }

    /// Turns the scored blocks into the per-key selection the attention reads.
    ///
    /// The scores have to be on the host for this: the ordering is
    /// (score descending, index ascending) over cells, every cell in a block
    /// shares its block's score, and the budget can cut a block in half. A
    /// GPU top-k that reproduces that tie-break exactly is the obvious next
    /// step; this is the version that is clearly correct.
    ///
    /// Returns nil when everything visible is kept, so the caller can skip
    /// the mask entirely rather than binding an all-ones buffer.
    func selectKeys(visibleKeys: Int) -> MTLBuffer? {
        guard visibleKeys > selectionWidth else { return nil }
        let keep = keepBuf.contents().bindMemory(to: UInt8.self, capacity: capacity)
        memset(keep, 0, visibleKeys)

        // The ragged tail of the query's own block is always kept: the
        // reference biases it above every score rather than ranking it.
        let completeCells = (visibleKeys / compressRatio) * compressRatio
        for cell in completeCells..<visibleKeys { keep[cell] = 1 }
        var remaining = selectionWidth - (visibleKeys - completeCells)
        guard remaining > 0 else { return keepBuf }

        let blocks = completeCells / compressRatio
        let scores = scoresBuf.contents().bindMemory(to: Float.self,
                                                     capacity: blocks + 1)
        // Descending score, ties to the lower block index -- which is what
        // the reference's stable cell ordering reduces to, because every cell
        // in a block carries the same score.
        let ranked = (0..<blocks).sorted {
            scores[$0] == scores[$1] ? $0 < $1 : scores[$0] > scores[$1]
        }
        for block in ranked {
            if remaining <= 0 { break }
            let take = min(compressRatio, remaining)
            let base = block * compressRatio
            for offset in 0..<take { keep[base + offset] = 1 }
            remaining -= take
        }
        return keepBuf
    }
}
