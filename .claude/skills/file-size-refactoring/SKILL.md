---
name: file-size-refactoring
description: This skill provides guidance for fixing file size violations detected by FileComplianceTests. Use when a C# file exceeds its line limit (800 lines default, or explicit limit for legacy files). The skill explains strategies to reduce file size while maintaining code quality.
---

# File Size Refactoring

This skill provides strategies for reducing file size when `FileComplianceTests` reports a violation.

## Why File Size Limits Exist

The limit exists for one reason: **a smaller file is easier for a human to understand, navigate, and maintain.** When a file grows beyond ~800 meaningful lines, it almost always means it has accumulated too many responsibilities. Hitting the limit is a healthy signal that the file deserves structural attention.

The correct response is always to **decompose the file into smaller, cohesive pieces** -- never to compress the existing code to squeeze under the limit.

## What the Test Measures

The test counts **non-empty lines** (excluding blank lines and whitespace-only lines). Comments, code, braces, usings -- all count. A file fails when it exceeds:
- **800 lines** (default limit for all files)
- **Explicit limit** for legacy files listed in `ExplicitFileLimits` in the test class

## The Only Valid Approaches

Regardless of whether a file is 5 lines or 500 lines over the limit, the approach is the same:

### 1. Remove Dead Code

Unused methods, commented-out code blocks, unreachable branches, unused usings -- remove them. This isn't about hitting a number; dead code hurts readability at any file size. This is general housekeeping.

### 2. Extract by Responsibility

Identify distinct responsibilities and move them to separate classes/files:
- Helper methods that don't depend on instance state -> static utility class
- Data transformation logic -> dedicated service class
- Validation logic -> separate validators
- Event handlers that form a subsystem -> separate handler class

### 3. Extract by Feature

Group related methods that implement a specific feature:
- All methods related to "mesh operations" -> `MeshOperations.cs`
- All methods related to "file import" -> `FileImportService.cs`
- All methods related to "undo/redo" -> `UndoRedoManager.cs`

### 4. Extract via Composition and Delegation

When methods are tightly coupled to instance state, that's a signal to extract a collaborator class that the original class *owns* and delegates to:
```csharp
// Before: one massive class
public class SceneContext
{
    // 200 lines of selection logic
    // 200 lines of undo/redo logic
    // 200 lines of core scene management
}

// After: separate responsibilities composed together
public class SceneContext
{
    private readonly SelectionManager _selection;
    private readonly UndoRedoManager _undoRedo;

    public SceneContext()
    {
        _selection = new SelectionManager(this);
        _undoRedo = new UndoRedoManager(this);
    }
}
```

This is usually preferable to a partial class because each extracted type is a real, nameable concept with its own testable surface area -- not just the same class spread across files (see the partial-class note under What NOT to Do).

### 5. Extract Interfaces and Implementations

When a class has multiple distinct interfaces:
```csharp
// Before: one large class
public class PrinterService
{
    // 200 lines of connection management
    // 200 lines of print job management
    // 200 lines of status monitoring
    // 200 lines of configuration
}

// After: interface-segregated classes
public class PrinterConnectionService : IPrinterConnection { ... }
public class PrintJobService : IPrintJobManager { ... }
public class PrinterStatusService : IPrinterStatus { ... }
public class PrinterConfigService : IPrinterConfig { ... }
```

## How to Evaluate an Extraction

Before splitting a file, ask:

- **Can you give the new file a clear, purposeful name?** If not, the split is probably artificial.
- **Does the extracted piece represent a cohesive concept?** It should make sense on its own, not just be "the second half of the file."
- **Would a new developer understand why this is its own file?** The structure should be self-evident.
- **Why did the file grow?** Understanding the pattern (feature creep, accumulated helpers, multiple responsibilities) leads to better decomposition than just looking for the biggest method to extract.

## C# Specific Patterns

- **Extension methods**: Move extension methods to their own static class files
- **Nested classes**: Extract nested classes to their own files
- **Constants/Enums**: Move large enum definitions or constant collections to dedicated files

## What NOT to Do

- **Don't remove blank lines, comments, or whitespace to shrink the file** -- This reduces readability, which is the exact opposite of the goal. The limit exists to *improve* readability, not to create pressure to sacrifice it.
- **Don't consolidate statements or compress code style** -- Multi-line formatting that aids readability should stay. Never trade clarity for line count.
- **Don't add to ExplicitFileLimits** -- That dictionary is only for freezing existing legacy files at their current size. Limits should only ever decrease.
- **Prefer extracting a type to adding a partial class** -- A partial class splits the file but not the responsibility: the class keeps its whole surface, and a member in one file can lean on private state in another without the reader seeing it. For a new split, extract a real, named type (an `internal static` class for a kernel algorithm whose functions already take their inputs as parameters costs almost nothing, and makes every cross-file call visible). A partial class is acceptable where extracting would force private state public or would be a redesign of a class that is doing too much; say why in a remark on the new file. Existing partial classes are not converted for their own sake - improve one when you are working in it for another reason. (Generated code, e.g. source generators, is always fine.)
- **Don't create artificial splits** -- Extracted classes should represent cohesive functionality, not arbitrary chunks. If you can't name it well, don't split it.
- **Don't treat small overages differently** -- A file at 801 lines needs the same structural thinking as one at 1200 lines. The extraction might be smaller, but the approach is the same.

## After Refactoring

1. Run `dotnet test --project ColmapSharp.Tests/ColmapSharp.Tests.csproj -- --treenode-filter "/*/*/FileComplianceTests/*"` to verify the fix
2. If a legacy file has been reduced, update `ExplicitFileLimits` in the test class to the new (lower) count
3. Remove the entry entirely when a legacy file reaches 800 lines or less
