import AppKit
import NVMAI
import NVMAIAppCore
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
            modelSection
            memorySection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var modelSection: some View {
        Section("Model") {
            LabeledContent("Path") {
                HStack(spacing: 6) {
                    Text(model.modelPathText)
                        .font(.caption)
                        .truncationMode(.middle)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                        .help(model.modelPathText)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.modelPathText, forType: .string)
                    } label: {
                        Label("Copy model path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy model path")
                }
            }
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
            }
            LabeledContent("State") {
                Text(model.presentation.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.requiresModelInstallation {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(model.installDescriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Installed size") {
                    Text(MetricFormat.storage(model.installDescriptor.installedBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let requirement = model.installRequirement {
                    LabeledContent("Available") {
                        Text(MetricFormat.storage(requirement.availableBytes))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    private var memorySection: some View {
        Section("Memory") {
            LabeledContent("Context scaling") {
                Picker("Context scaling", selection: $model.runtimeOptions.ropeScalingMode) {
                    Text("None").tag(RuntimeRoPEScalingMode.none)
                    Text("YaRN").tag(RuntimeRoPEScalingMode.yarn)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .onChange(of: model.runtimeOptions.ropeScalingMode) { _, mode in
                    model.maxContextTokens = mode == .yarn
                        ? RuntimeConfiguration.defaultYaRNContextTokens
                        : AppContextLengthOption.twoFiftySixK.tokens
                }
            }
            LabeledContent("Context") {
                Picker("Context", selection: $model.maxContextTokens) {
                    ForEach(contextOptions) { option in
                        Text(option.menuLabel(
                            precision: model.runtimeOptions.kvCachePrecision))
                            .tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            LabeledContent("KV cache") {
                Picker("KV cache", selection: $model.runtimeOptions.kvCachePrecision) {
                    ForEach([KVCachePrecision.fp16, .int8, .int4], id: \.rawValue) { precision in
                        Text(precision.label).tag(precision)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }
            LabeledContent("Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach([AppRuntimeOptions.automaticSlotCount] + AppRuntimeOptions.allowedSlotCounts,
                            id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
            }
            Text("Auto sizes the expert cache from the model's tuning profile (the measured optimum for this model and quantization on this Mac). Quantized KV reduces long-context memory. YaRN extends the model beyond its native 256K context; 1M is selected by default when enabled. These settings apply after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var contextOptions: [AppContextLengthOption] {
        model.runtimeOptions.ropeScalingMode == .yarn
            ? AppContextLengthOption.yarnCases
            : AppContextLengthOption.nativeCases
    }

    private var generationSection: some View {
        Section("Generation") {
            LabeledContent("Thinking") {
                Picker("Thinking", selection: $model.runtimeOptions.thinkingMode) {
                    Text("Off").tag(ModelThinkingMode.off)
                    Text("On").tag(ModelThinkingMode.on)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
            }
            Text("Ornith and Qwen expose a binary thinking switch; they do not define Low, Medium, or High effort levels. This setting applies after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Model's sampling defaults", isOn: $model.samplingFollowsModel)
                .toggleStyle(.switch)
            Text("Temperature, Top-K and Top-P from the model's tuning profile. Switch off to set your own; editing a value below switches it off too.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                    .fixedSize()
                }
            }
            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .disabled(!model.topKEnabled)
            if model.topKEnabled && model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
            if model.runtimeOptions.prefillEnabled {
                LabeledContent("Prefill chunk") {
                    Picker("Prefill chunk", selection: $model.runtimeOptions.prefillChunkTokens) {
                        ForEach(AppRuntimeOptions.allowedPrefillChunkTokens, id: \.self) { tokens in
                            Text("\(tokens)").tag(tokens)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .fixedSize()
                }
                Text("Larger chunks reduce repeated expert-file reads for long prompts, but use more temporary GPU memory.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 8) {
                Text("RDADVISE")
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

}
