import NVMAIAppCore
import NVMAIMacPresentation
import SwiftUI

/// Shown when the model's payload is complete but its install receipt was
/// issued for a different directory — i.e. the model was moved or renamed.
///
/// This state exists so the app does not answer a moved model with the
/// installer. Re-attesting re-hashes what is already on disk and rebinds the
/// receipt; the alternative the app used to offer was a multi-gigabyte
/// re-download of a model the user already has.
struct ModelReattestationView: View {
    let model: AppModel

    var body: some View {
        ScrollView {
            VStack(spacing: 22) {
                identity
                explanation
                actions
            }
            .frame(maxWidth: 560)
            .padding(.horizontal, 28)
            .padding(.vertical, 48)
            .frame(maxWidth: .infinity)
        }
    }

    private var identity: some View {
        VStack(spacing: 10) {
            Image(systemName: "externaldrive.badge.checkmark")
                .font(.system(.largeTitle, design: .rounded))
                .foregroundStyle(NVMAIMacTheme.accentColor)
            Text("Model needs re-verifying")
                .font(.system(.title2, design: .rounded).weight(.semibold))
            Text(model.modelPathText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    private var explanation: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("The model files are all here and match their manifest. Only the "
                 + "install receipt still points at the folder the model used to "
                 + "live in, which happens when it is moved or renamed.")
            Text("Re-verifying re-reads the files already on disk and re-issues the "
                 + "receipt. Nothing is downloaded.")
                .foregroundStyle(.secondary)
            if let reason = reattestationReason {
                Text(reason)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .font(.system(.body, design: .rounded))
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(18)
        .background {
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
    }

    private var actions: some View {
        VStack(spacing: 10) {
            Button {
                model.reattestModel()
            } label: {
                Text(model.canReattestModel ? "Re-verify Model" : "Re-verifying…")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!model.canReattestModel)

            if let error = model.error {
                Text(error.localizedDescription)
                    .font(.system(.caption, design: .rounded))
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
    }

    private var reattestationReason: String? {
        if case .needsReattestation(let reason) = model.installationStatus { return reason }
        return nil
    }
}
