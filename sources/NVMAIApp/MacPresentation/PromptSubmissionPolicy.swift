import SwiftUI
import NVMAIAppCore

public enum PromptSubmissionDecision: Equatable, Sendable {
    case submit
    case consume
    case deferToEditor
}

public enum PromptSubmissionPolicy {
    public static func decision(
        newlineShortcut: AppNewlineShortcut,
        modifiers: EventModifiers,
        canRun: Bool,
        hasMarkedText: Bool,
        isRepeat: Bool = false
    ) -> PromptSubmissionDecision {
        let nativeModifiers: EventModifiers = [.command, .shift, .option, .control]

        // `.return` is newline mode: Return never submits, so it always defers
        // to the editor (insert a newline), regardless of modifiers.
        guard newlineShortcut == .shiftReturn else { return .deferToEditor }

        // D27: Command+Return is the generate shortcut. When the generate
        // action is unavailable the button is disabled, the shortcut does not
        // fire, and the keypress reaches the editor — consume it rather than
        // letting TextEditor insert a newline.
        if modifiers.contains(.command), !canRun {
            return .consume
        }

        guard modifiers.intersection(nativeModifiers).isEmpty else {
            return .deferToEditor
        }
        if isRepeat { return .consume }
        guard !hasMarkedText else { return .deferToEditor }
        return canRun ? .submit : .consume
    }
}
