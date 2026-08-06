import Foundation

enum GTurboLayoutValidator {
    static func validate(path: String, plan: RepackPlan) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let layers = root["layers"] as? [[String: Any]] else {
            throw RepackError.configurationInvalid(detail: "layout.json validation failed: malformed root")
        }
        for layerObj in layers {
            guard let layerIndex = layerObj["layer"] as? Int,
                  let experts = layerObj["experts"] as? [[String: Any]],
                  let planLayer = plan.layers.first(where: { $0.layerIndex == layerIndex }) else {
                throw RepackError.configurationInvalid(detail: "layout.json validation failed: malformed layer")
            }
            if planLayer.expertsPerLayer == 0 { continue }
            var seenLogical = Set<Int>()
            var seenOffsets = Set<UInt64>()
            for expertObj in experts {
                guard let expert = expertObj["expert"] as? Int,
                      let offset = (expertObj["offset"] as? NSNumber)?.uint64Value,
                      let size = (expertObj["size"] as? NSNumber)?.uint64Value else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: malformed expert")
                }
                guard expert >= 0 && expert < planLayer.expertsPerLayer else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: expert out of range")
                }
                guard seenLogical.insert(expert).inserted else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: duplicate expert \(expert)")
                }
                guard seenOffsets.insert(offset).inserted else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: duplicate offset \(offset)")
                }
                guard offset % Layout.pageBytes == 0 else {
                    throw RepackError.configurationInvalid(detail: "layout.json validation failed: unaligned offset \(offset)")
                }
                let expected = UInt64(expert) * planLayer.expertStride
                guard offset == expected else {
                    throw RepackError.configurationInvalid(detail:
                        "layout.json validation failed: offset \(offset) != expert * stride \(expected)")
                }
                guard size == planLayer.expertStride,
                      offset + size <= planLayer.fileSize else {
                    throw RepackError.configurationInvalid(detail:
                        "layout.json validation failed: expert range outside layer file")
                }
                // Every per-tensor sub-dict must agree with the plan slice it
                // encodes: offset/size inside the expert blob, dtype, logical
                // shape and quantization bits.
                guard let tensors = expertObj["tensors"] as? [String: Any] else {
                    throw RepackError.configurationInvalid(detail:
                        "layout.json validation failed: expert \(expert) has no tensors dict")
                }
                for slice in planLayer.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    guard let sub = tensors[key] as? [String: Any] else {
                        throw RepackError.configurationInvalid(detail:
                            "layout.json validation failed: expert \(expert) missing tensor \(key)")
                    }
                    guard let subOffset = (sub["offset"] as? NSNumber)?.uint64Value,
                          let subSize = (sub["size"] as? NSNumber)?.uint64Value else {
                        throw RepackError.configurationInvalid(detail:
                            "layout.json validation failed: tensor \(key) has no offset/size")
                    }
                    guard subOffset == slice.offsetInExpertBlob,
                          subSize == slice.sizeInExpertBlob,
                          subOffset <= size,
                          subSize <= size - subOffset else {
                        throw RepackError.configurationInvalid(detail:
                            "layout.json validation failed: tensor \(key) range does not match the plan or exceeds the expert blob")
                    }
                    guard let dtype = sub["dtype"] as? String,
                          (dtype == "U32") == (slice.dtype == 0),
                          (dtype == "BF16") == (slice.dtype == 1) else {
                        throw RepackError.configurationInvalid(detail:
                            "layout.json validation failed: tensor \(key) dtype does not match the plan")
                    }
                    guard let shape = sub["shape"] as? [Any],
                          shape.count == slice.logicalShape.count,
                          zip(shape, slice.logicalShape).allSatisfy({ pair in
                              (pair.0 as? NSNumber)?.uint64Value == pair.1
                          }) else {
                        throw RepackError.configurationInvalid(detail:
                            "layout.json validation failed: tensor \(key) shape does not match the plan")
                    }
                    let bits = (sub["bits"] as? NSNumber)?.intValue
                    guard bits == slice.bitsForWeights else {
                        throw RepackError.configurationInvalid(detail:
                            "layout.json validation failed: tensor \(key) bits do not match the plan")
                    }
                }
                guard tensors.count == planLayer.subTensors.count else {
                    throw RepackError.configurationInvalid(detail:
                        "layout.json validation failed: expert \(expert) has unexpected tensor keys")
                }
            }
            guard seenLogical.count == planLayer.expertsPerLayer else {
                throw RepackError.configurationInvalid(detail:
                    "layout.json validation failed: missing experts in layer \(layerIndex)")
            }
        }
    }
}
