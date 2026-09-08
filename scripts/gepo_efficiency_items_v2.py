"""77 additional EFFICIENCY driver items, mined 2026-09-08.

HOW THESE DIFFER FROM THE FIRST 51
----------------------------------
The original block contrasted an efficient MODEL GROUP against an inefficient one,
and had to argue the identity confound away. These items are built on a stronger
design that removes it: every run on BOTH sides of every split below reached the
SAME VERIFIED OUTCOME -- the oracle returned FIXED. The groups differ only in what
it cost them.

    corpus            184 manic_miner runs with recoverable event logs, 20,846 actions
    both groups       verdict = FIXED (the oracle passed)
    CHEAP-SUCCESS     46 runs, median  22 actions (range 12-28)
    EXPENSIVE-SUCCESS 46 runs, median 144 actions (range 58-751)
    cost ratio        6.7x more actions for the same verified result

Both terciles are model-mixed by construction -- CHEAP contains omnimerge-v4 x29,
v6 x7, qwen36-base x4, jprimep3 x2, qwen3.8 x2; EXPENSIVE contains v6 x9, coderx x8,
a3b-coder x7, omnimerge-v4 x5, gepo3 x3, gepo1 x3. omnimerge-v4 and v6 appear on BOTH
sides, so no item below can be answered by knowing which model produced it. That is
the point: this is "both worked, one was better", not "good model vs bad model".

Every `evidence` string carries the observed conditional split and its sample size.
"""

ITEMS_V2 = []


def M2(domain, principle, q, correct, wrong, evidence):
    ITEMS_V2.append((domain, principle, q, correct, wrong, evidence))


# ---------------------------------------------------------------------------
# D1  RE-RUNNING WITHOUT HAVING CHANGED ANYTHING  (the largest divergence found)
# After running the program, the expensive successes ran it AGAIN 49.6% of the
# time; the cheap successes did so 13.9%. Same outcome, 3.6x the null re-runs.
# ---------------------------------------------------------------------------
E1 = ("measured on 3,067 post-run transitions across 92 runs that ALL reached the "
      "verified fix: the expensive group ran the program again immediately 49.6% of "
      "the time against the cheap group's 13.9%")

M2("general", "rerun-needs-a-change",
   "You have just run the failing program and read its error output. You have not "
   "edited anything since. What is the most economical next action?",
   "Use the error to decide what to change, and change it",
   ["Run the program again to be sure the failure is reproducible",
    "Run the program a third time to check the error is not intermittent",
    "Re-read the error output more carefully before doing anything else"], E1)

M2("testing", "rerun-needs-a-change",
   "A test run has just finished and reported three failures. Nothing in the "
   "repository has been modified since. What should you do next?",
   "Pick one failure and change the code or test it points at",
   ["Re-run the suite to confirm the same three fail",
    "Re-run only the first failing test on its own to see it fail again",
    "Run the suite with more verbose output and compare the two runs"], E1)

M2("build", "rerun-needs-a-change",
   "A build has just failed with a compiler error naming a specific file and line. "
   "You have made no edits since. What is the most economical next action?",
   "Open that file at that line and fix what the compiler objected to",
   ["Run the build again in case the failure was a stale artefact",
    "Run a clean build to rule out incremental-build noise",
    "Run the build with higher verbosity to see the same error in more detail"], E1)

M2("data-pipeline", "rerun-needs-a-change",
   "A data job has just aborted with a schema-mismatch error identifying the offending "
   "column. Nothing has been altered since. What next?",
   "Change the schema or the column mapping the error identified",
   ["Revert the last schema migration and run the job again",
    "Re-run the job on a smaller input slice to watch it fail faster",
    "Re-run with debug logging enabled and diff the two failures"], E1)

# ---------------------------------------------------------------------------
# D2  REPRODUCE BEFORE SURVEYING  (confirmed on outcome-matched runs)
# Opening action: cheap successes ran the program 52.2% of the time (24 of 46);
# expensive successes did so 17.4% (8 of 46) and opened by READING 60.9% vs 37.0%.
# ---------------------------------------------------------------------------
E2 = ("first action of 92 runs that all reached the verified fix: the cheap group "
      "ran the program 52.2% of the time (24 of 46) against the expensive group's "
      "17.4% (8 of 46), while the expensive group opened by reading 60.9% vs 37.0%")

M2("general", "reproduce-first",
   "You are handed a defect report for a program you have not seen before. The "
   "repository is checked out and the program is runnable. What do you do first?",
   "Run the program and watch it fail for yourself",
   ["Rewrite the module the report names, since it is implicated",
    "List the project's files to build a mental map before touching anything",
    "Read the README and the architecture notes so you know the conventions"], E2)

M2("mobile", "reproduce-first",
   "Users report an app crashing on launch on one device family. You have the source "
   "and a matching device. What is the most economical first action?",
   "Launch the app on that device and capture the crash",
   ["Roll back the most recent release and see whether crashes stop",
    "Review the last month of commits touching startup",
    "Audit the manifest and permissions for that device family"], E2)

M2("performance", "reproduce-first",
   "A batch job is reported as 'much slower since Tuesday'. You have the job and a "
   "representative input. What first?",
   "Run it on that input and measure where the time goes",
   ["Add caching to the paths you assume are hot, then compare",
    "Compare Tuesday's deploy against the previous one line by line",
    "Survey the data volumes to see whether the input simply grew"], E2)

M2("web-service", "reproduce-first",
   "An endpoint intermittently returns malformed JSON. You can call it yourself. "
   "What is the most economical first action?",
   "Call it until you have one malformed response in hand",
   ["Wrap the endpoint in a validator that rejects malformed output",
    "Study the request-handling middleware chain end to end",
    "Review the schema definitions for optional fields"], E2)

# ---------------------------------------------------------------------------
# D3  A READ SHOULD PRODUCE A CHANGE
# After reading a file, the cheap successes edited next 59.0% of the time; the
# expensive successes 36.8%, preferring to run (28.3% vs 14.7%) or read again.
# ---------------------------------------------------------------------------
E3 = ("measured on 2,147 post-read transitions across 92 runs that all reached the "
      "verified fix: the cheap group edited next 59.0% of the time against the "
      "expensive group's 36.8%, which instead ran the program 28.3% vs 14.7%")

# NOTE: an earlier draft of this item duplicated the v1 row of the same name
# (same domain, same correct answer, two shared distractors). Replaced with a
# distinct scenario; near-duplicate detection over the built pool caught it.
M2("mobile", "reads-should-produce-changes",
   "You have read the lifecycle callback and found it releases a resource the next "
   "screen still needs. That accounts for the crash. What is the most economical next "
   "action?",
   "Change the callback so it stops releasing that resource",
   ["Rewrite the lifecycle callback from scratch to be safe",
    "Read the next screen to see exactly how it uses the resource",
    "Read the other lifecycle callbacks to see if they do the same"], E3)

M2("web-service", "reads-should-produce-changes",
   "You have read the handler and found the branch that returns the wrong status code. "
   "What next?",
   "Correct the branch that returns the wrong status code",
   ["Make every branch in the handler return the same status code",
    "Read the router to see how the handler is reached",
    "Read the other handlers to see how they set status codes"], E3)

M2("data-pipeline", "reads-should-produce-changes",
   "You have read the transform and located the line that drops rows with a null key. "
   "That is the reported defect. What next?",
   "Change the line that drops the null-key rows",
   ["Run the pipeline again to see the row count drop",
    "Read the loader upstream to check where the nulls come from",
    "Read the downstream consumer to see whether it tolerates nulls"], E3)

M2("testing", "reads-should-produce-changes",
   "You have read the failing assertion and understand why it fires. What is the most "
   "economical next action?",
   "Fix the code or the assertion, whichever is wrong",
   ["Run the test again to observe the assertion fail",
    "Read the fixtures to understand how the input was built",
    "Read neighbouring tests to see whether they share the assumption"], E3)

# ---------------------------------------------------------------------------
# D4  FINISH THE CHECK SWEEP  (cheap runs batch their verification)
# After a check, cheap successes checked again 28.0% of the time; expensive ones
# 8.6%, jumping instead to running (35.1% vs 17.0%) or reading (29.6% vs 13.0%).
# ---------------------------------------------------------------------------
E4 = ("measured on 448 post-check transitions across 92 runs that all reached the "
      "verified fix: the cheap group ran a further check 28.0% of the time against "
      "the expensive group's 8.6%, which instead jumped to running the program "
      "(35.1% vs 17.0%) or back to reading (29.6% vs 13.0%)")

M2("general", "finish-the-check-sweep",
   "You have just validated one of the four files you edited and it is clean. What is "
   "the most economical next action?",
   "Validate the other three now, while you are set up to",
   ["Run the whole program to see whether the four edits worked",
    "Re-read the first file to be sure the validation was meaningful",
    "Move on and let the next full run surface any remaining problem"], E4)

M2("build", "finish-the-check-sweep",
   "You have syntax-checked one of several changed source files and it passes. What "
   "next?",
   "Syntax-check the rest of the changed files",
   ["Commit the file you checked and open a pull request for it",
    "Read the checked file again to confirm the change is what you meant",
    "Commit the checked file and handle the others separately"], E4)

M2("data-pipeline", "finish-the-check-sweep",
   "You have validated one of five modified config files against its schema and it is "
   "valid. What is most economical?",
   "Validate the remaining four against their schemas",
   ["Run the pipeline end to end and see whether it starts",
    "Re-open the validated file to double-check the schema was the right one",
    "Deploy and rely on the pipeline's own startup validation"], E4)

M2("testing", "finish-the-check-sweep",
   "You have run one of the tests you expect your change to affect and it passes. What "
   "next?",
   "Run the other tests your change affects",
   ["Run the entire suite from the beginning",
    "Read your change again to reassure yourself it is correct",
    "Consider the change verified and move to the next task"], E4)

# ---------------------------------------------------------------------------
# D5  ACT ON THE CHECK RESULT INSTEAD OF RE-ACQUIRING IT
# After a check, the expensive successes went back to READING 29.6% of the time
# against the cheap group's 13.0% -- re-acquiring context the check just gave them.
# ---------------------------------------------------------------------------
E5 = ("measured on 448 post-check transitions across 92 runs that all reached the "
      "verified fix: the expensive group returned to reading 29.6% of the time "
      "against the cheap group's 13.0%")

M2("general", "act-on-the-result",
   "A validation step has just told you exactly which line is malformed. What is the "
   "most economical next action?",
   "Go to the line the validator named and fix it",
   ["Read the whole file to understand the context around it",
    "Read the validator's documentation to be sure you read the message correctly",
    "Run the validation again to see whether the message changes"], E5)

M2("web-service", "act-on-the-result",
   "A schema check reports that one response field has the wrong type. What next?",
   "Change that field's type where it is produced",
   ["Read the full schema to see what else might be wrong",
    "Read the serialiser end to end to understand how types are assigned",
    "Run the schema check again with stricter settings"], E5)

M2("build", "act-on-the-result",
   "A linter has named one file and one rule violation. What is most economical?",
   "Fix the violation the linter named, in the file it named",
   ["Read the file from the top to see whether other rules are also violated",
    "Read the linter's rule documentation before deciding",
    "Run the linter over the whole project to gather every violation first"], E5)

M2("mobile", "act-on-the-result",
   "A layout inspector has identified the constraint causing the overlap. What next?",
   "Change the constraint the inspector identified",
   ["Rebuild the screen with a different layout container entirely",
    "Read the other screens to see how they avoid this",
    "Re-run the inspector on a different device size first"], E5)

# ---------------------------------------------------------------------------
# D6  NO SCRATCH PROBES  (the clearest single behaviour in the expensive group)
# After an edit, expensive successes ran some OTHER command -- a scratch script, an
# ad-hoc probe -- 15.1% of the time against the cheap group's 1.3%. 11.6x.
# ---------------------------------------------------------------------------
E6 = ("measured on 2,642 post-edit transitions across 92 runs that all reached the "
      "verified fix: the expensive group ran an ad-hoc scratch command 15.1% of the "
      "time against the cheap group's 1.3% -- an 11.6x difference -- while the cheap "
      "group ran the actual program (32.4% vs 26.1%) or a real check (12.5% vs 4.2%)")

M2("general", "no-scratch-probes",
   "You have just made an edit you believe fixes the defect. You want to know whether "
   "it worked. What is the most economical next action?",
   "Run the program the way the defect report runs it",
   ["Write a small script that exercises just the function you changed",
    "Write a script that prints the intermediate values so you can inspect them",
    "Write a one-off harness that isolates the module from its dependencies"], E6)

M2("testing", "no-scratch-probes",
   "You have changed a function that an existing test already covers. How do you check "
   "the change?",
   "Run the existing test that already covers the function",
   ["Write a small script calling the function with a sample input",
    "Write a scratch file that prints the function's output for several inputs",
    "Build a temporary harness so you can call it without the test framework"], E6)

M2("data-pipeline", "no-scratch-probes",
   "You have edited a transform step. The pipeline can be run on a sample input. How do "
   "you verify the edit?",
   "Run the pipeline on the sample input",
   ["Write a scratch script that calls the transform directly",
    "Write a script that dumps the intermediate dataframe for inspection",
    "Write a standalone copy of the transform to experiment with"], E6)

M2("performance", "no-scratch-probes",
   "You have applied an optimisation to a slow function. How do you find out whether it "
   "helped?",
   "Run the existing benchmark that measures it",
   ["Write a small timing script around the function",
    "Write a script that runs the function in a loop and prints the mean",
    "Build an isolated micro-benchmark outside the project"], E6)

# ---------------------------------------------------------------------------
# D7  READ THE FAILURE, DON'T RE-TRIGGER IT
# After running the program, the cheap successes read next 32.5% of the time and
# searched 12.4%; the expensive ones read 20.1% and searched 1.8%, preferring to run.
# ---------------------------------------------------------------------------
E7 = ("measured on 3,067 post-run transitions across 92 runs that all reached the "
      "verified fix: the cheap group moved from running to reading 32.5% of the time "
      "(vs 20.1%) and to searching 12.4% (vs 1.8%), while the expensive group ran "
      "again 49.6% of the time")

M2("general", "locate-from-the-error",
   "The program has failed and printed a stack trace naming a function you have not "
   "looked at. What is the most economical next action?",
   "Open the function the stack trace named",
   ["Run the program again to see whether the trace is stable",
    "Search the codebase for every call site of every frame in the trace",
    "Read the entry point and follow the control flow down to it"], E7)

M2("web-service", "locate-from-the-error",
   "A request has failed and the log line contains a distinctive error string. You do "
   "not know where it is raised. What next?",
   "Search the codebase for that distinctive error string",
   ["Add a catch-all handler so the request stops failing outright",
    "Read the handler chain from the router down",
    "Enable debug logging everywhere and repeat the request"], E7)

M2("build", "locate-from-the-error",
   "The build failed with an error mentioning a symbol you do not recognise. What is "
   "most economical?",
   "Search the codebase for that symbol's definition",
   ["Run the build again with a clean cache",
    "Read the build configuration from the top",
    "Read each source file the build touched in order"], E7)

M2("data-pipeline", "locate-from-the-error",
   "The job aborted citing a named validation rule. You do not know where the rule is "
   "defined. What next?",
   "Search for the rule name the abort message cited",
   ["Disable the validation rule so the job completes",
    "Read the validation framework's source to understand rule dispatch",
    "Read every rule definition file in order until you find it"], E7)

# ---------------------------------------------------------------------------
# D8  A SEARCH SHOULD NARROW, NOT REPEAT
# After a search, expensive successes searched AGAIN 26.3% of the time against the
# cheap group's 15.2%; the cheap group edited next 28.8% against 15.8%.
# ---------------------------------------------------------------------------
E8 = ("measured on 237 post-search transitions across 92 runs that all reached the "
      "verified fix: the expensive group searched again 26.3% of the time against the "
      "cheap group's 15.2%, while the cheap group went on to edit 28.8% vs 15.8%")

M2("general", "no-search-churn",
   "Your search returned one clear match in one file. What is the most economical next "
   "action?",
   "Open the single match your search returned and act on it",
   ["Change every file the broader pattern would have matched",
    "Search for related identifiers to build a fuller picture",
    "Search the test directory for the same term before deciding"], E8)

M2("testing", "no-search-churn",
   "Searching for the failing helper's name returned exactly one definition. What next?",
   "Open the one definition the search returned and work from it",
   ["Rewrite the helper and every test that calls it together",
    "Search for every usage before reading the definition",
    "Search the fixtures for the same name first"], E8)

M2("web-service", "no-search-churn",
   "A search for the error string found the single place it is raised. What is most "
   "economical?",
   "Read the single raise site and change what is wrong there",
   ["Replace the error with a generic message so it stops appearing",
    "Search for the error's identifier in the client code too",
    "Search the logs for other occurrences before touching anything"], E8)

M2("performance", "no-search-churn",
   "Profiling named one function and a search located it immediately. What next?",
   "Open the function profiling named and address what makes it slow",
   ["Search for other functions with similar names for comparison",
    "Search for every caller before reading the function",
    "Search for previous optimisation attempts in the history first"], E8)

# ---------------------------------------------------------------------------
# D9  VERIFY THE CHANGE YOU JUST MADE
# After an edit, cheap successes ran a real check 12.5% of the time against 4.2%,
# and ran the program 32.4% vs 26.1%. The expensive group read again instead.
# ---------------------------------------------------------------------------
E9 = ("measured on 2,642 post-edit transitions across 92 runs that all reached the "
      "verified fix: the cheap group checked the edit 12.5% of the time against the "
      "expensive group's 4.2%, and ran the program 32.4% vs 26.1%, while the expensive "
      "group returned to reading 24.5% vs 20.2%")

M2("general", "verify-each-change",
   "You have made one edit. You have three more edits planned. What is the most "
   "economical next action?",
   "Check that this edit is correct before making the next",
   ["Make all three remaining edits, then check everything at once",
    "Re-read the file you just edited to confirm it looks right",
    "Read the next file you intend to edit"], E9)

M2("build", "verify-each-change",
   "You have changed one source file in a project that takes several minutes to build "
   "fully. There is a fast syntax check available. What next?",
   "Run the fast check on the file you changed",
   ["Start the full build now and use the time to plan the next edit",
    "Make the remaining planned edits first, then build once",
    "Re-read the change to satisfy yourself it compiles"], E9)

M2("mobile", "verify-each-change",
   "You have modified one screen's layout. Rebuilding and deploying to the device takes "
   "a while, but a layout preview is instant. What is most economical?",
   "Look at the preview for that screen",
   ["Deploy to the device and check it there",
    "Make the other planned layout changes first, then deploy once",
    "Re-read the layout file to confirm the change"], E9)

M2("data-pipeline", "verify-each-change",
   "You have edited one transform among several you intend to change. A cheap unit "
   "check exists for it. What next?",
   "Run the cheap unit check on the transform you just edited",
   ["Edit the remaining transforms and run the full pipeline once",
    "Re-read the transform to be sure the edit is right",
    "Run the whole pipeline to see the effect in context"], E9)

# ---------------------------------------------------------------------------
# D10  DO NOT RE-READ WHAT YOU ALREADY HOLD
# The expensive successes followed a read with another read 21.6% of the time
# against the cheap group's 12.4%.
# ---------------------------------------------------------------------------
E10 = ("measured on 2,147 post-read transitions across 92 runs that all reached the "
       "verified fix: the expensive group read again immediately 21.6% of the time "
       "against the cheap group's 12.4%")

M2("general", "no-reacquisition",
   "You read a file two actions ago and nothing has changed it since. You need a detail "
   "from it. What is the most economical action?",
   "Use the timeout value you have already read",
   ["Change the detail everywhere it could plausibly appear",
    "Read the file again, this time the whole of it",
    "Read a neighbouring file to cross-check the detail"], E10)

M2("web-service", "no-reacquisition",
   "You have already read the config that sets the timeout, and nothing has modified it. "
   "You now need the timeout value. What next?",
   "Use the timeout value you already read from the config",
   ["Raise the timeout in every config you can find",
    "Re-open the config and also read the defaults it inherits",
    "Search for other places the timeout might be overridden first"], E10)

M2("testing", "no-reacquisition",
   "You read the fixture file earlier in this session. It has not been touched. You need "
   "to know one of its fields. What is most economical?",
   "Rely on the field you already read from the fixture",
   ["Read the fixture again to be safe",
    "Read the fixture and its factory to be thorough",
    "Re-run the test to see the fixture printed"], E10)

M2("data-pipeline", "no-reacquisition",
   "You have already read the schema definition and nothing has changed it. You need one "
   "column's type. What next?",
   "Use the column type you already read from the schema",
   ["Re-read the schema to confirm",
    "Re-read the schema and the migration history around it",
    "Query the live table to check the type independently"], E10)

# ---------------------------------------------------------------------------
# D11  RECORD PROGRESS INSTEAD OF RE-DERIVING IT
# After a check, cheap successes recorded progress 12.0% of the time against 2.0%.
# ---------------------------------------------------------------------------
E11 = ("measured on 448 post-check transitions across 92 runs that all reached the "
       "verified fix: the cheap group recorded its progress 12.0% of the time against "
       "the expensive group's 2.0%")

M2("general", "record-progress",
   "You have just confirmed that two of the five suspected causes are ruled out. What is "
   "the most economical next action?",
   "Note which two are eliminated, then move to the third",
   ["Apply a speculative fix to all five causes at once",
    "Start again from the first cause with a different method",
    "Investigate all three remaining causes at once"], E11)

M2("testing", "record-progress",
   "You have established which three of twelve failing tests share a single root cause. "
   "What next?",
   "Write that grouping down and fix the shared cause",
   ["Patch each of the twelve tests individually to make them pass",
    "Investigate each of the twelve independently to be safe",
    "Re-derive the grouping from the failure messages once more"], E11)

M2("web-service", "record-progress",
   "You have determined that the fault occurs only on requests carrying a particular "
   "header. What is most economical?",
   "Record that the fault needs that header, and work from that",
   ["Re-test with and without the header to be sure",
    "Test every other header combination for completeness",
    "Re-derive the condition from the logs a second time"], E11)

M2("build", "record-progress",
   "You have narrowed a build break to one of two modules. What next?",
   "Note the two candidates and test the more likely one",
   ["Re-run the build to confirm the narrowing",
    "Widen the search again to be sure you did not exclude too early",
    "Re-derive the narrowing from the build log once more"], E11)

# ---------------------------------------------------------------------------
# D12  TRUST A PASSING CHECK
# After a check, expensive successes ran the program 35.1% of the time against the
# cheap group's 17.0% -- re-establishing by an expensive route what they just learned.
# ---------------------------------------------------------------------------
E12 = ("measured on 448 post-check transitions across 92 runs that all reached the "
       "verified fix: the expensive group ran the full program 35.1% of the time "
       "against the cheap group's 17.0%")

M2("general", "trust-the-check",
   "A targeted check has just confirmed that the file you edited is well formed. You "
   "have two more files to edit. What is the most economical next action?",
   "Edit the next file; the targeted check already passed",
   ["Run the whole program to confirm nothing else broke",
    "Run the program and then re-check the same file",
    "Re-read the edited file to confirm the check was meaningful"], E12)

M2("build", "trust-the-check",
   "A fast type-check over your changed files passes. The full build takes ten minutes. "
   "You still have edits to make. What next?",
   "Make the remaining edits; the type-check already passed",
   ["Start the full build now to be certain",
    "Run the full build and re-run the type-check afterwards",
    "Re-read the changed files before continuing"], E12)

M2("data-pipeline", "trust-the-check",
   "Schema validation of your edited config passes. The full pipeline run takes an hour. "
   "What is most economical?",
   "Continue with the remaining work; validation already passed",
   ["Start a full pipeline run to confirm",
    "Run the pipeline and validate again afterwards",
    "Re-read the config to be sure the validator saw your change"], E12)

M2("testing", "trust-the-check",
   "The single test covering your change passes. The full suite takes twenty minutes and "
   "you have more edits planned. What next?",
   "Make the next edit; the covering test already passed",
   ["Run the full suite now to be safe",
    "Run the full suite and then re-run the single test",
    "Re-read your change before continuing"], E12)

# ---------------------------------------------------------------------------
# D13  CHECK AFTER RUNNING, NOT ANOTHER RUN
# After running the program, cheap successes ran a targeted check 17.2% of the time
# against the expensive group's 3.3% -- 5.2x.
# ---------------------------------------------------------------------------
E13 = ("measured on 3,067 post-run transitions across 92 runs that all reached the "
       "verified fix: the cheap group followed a run with a targeted check 17.2% of "
       "the time against the expensive group's 3.3%")

M2("general", "check-after-run",
   "The program ran and failed in a way suggesting a malformed file rather than wrong "
   "logic. What is the most economical next action?",
   "Run the targeted check that would confirm the file is malformed",
   ["Run the program again with more output",
    "Read the file from the beginning looking for the problem",
    "Run the program under a debugger"], E13)

M2("web-service", "check-after-run",
   "A request failed with an error that looks like invalid configuration rather than a "
   "code fault. What next?",
   "Run the configuration validator against it",
   ["Restart the service and change the configuration at the same time",
    "Read the configuration file line by line",
    "Restart the service and try once more"], E13)

M2("build", "check-after-run",
   "A build failed with an error suggesting a malformed manifest rather than a code "
   "problem. What is most economical?",
   "Run the validator against the manifest it complained about",
   ["Run the build again and read more of the output",
    "Read the manifest from the top",
    "Clean the build directory and rebuild"], E13)

M2("mobile", "check-after-run",
   "The app failed to install with a message hinting at a malformed resource file. What "
   "next?",
   "Run the resource validator against the file named",
   ["Attempt the install again to see the message once more",
    "Read the resource file looking for the fault",
    "Rebuild the package and retry the install"], E13)

# ---------------------------------------------------------------------------
# D14  ONE THING AT A TIME  (economy of the whole trajectory, not one step)
# The two groups reached the SAME verified outcome; the expensive one used 6.7x the
# actions -- median 144 against 22.
# ---------------------------------------------------------------------------
E14 = ("92 runs that all reached the same verified fix: the cheap group used a median "
       "of 22 actions (range 12-28), the expensive group 144 (range 58-751) -- 6.7x "
       "for an identical outcome")

M2("general", "act-early",
   "You have a plausible hypothesis about the defect and a cheap way to test it. You "
   "could also gather more context first. What is the most economical action?",
   "Test the hypothesis now, using the cheap test you have",
   ["Apply the fix you would make if the hypothesis were confirmed",
    "Enumerate the alternative hypotheses before testing any",
    "Read the surrounding code so you can interpret the result"], E14)

M2("performance", "act-early",
   "You suspect one query is responsible for a slowdown, and you can time it directly. "
   "What next?",
   "Time that query directly and see what it costs",
   ["Profile the whole request path first for context",
    "List every query the path issues before measuring any",
    "Read the query planner documentation before measuring"], E14)

M2("testing", "act-early",
   "You believe a single fixture change will fix eight failing tests, and running them "
   "takes seconds. What is most economical?",
   "Make the change and run the eight",
   ["Read all eight tests first to confirm they share the cause",
    "Run the eight again to re-confirm they fail the same way",
    "Investigate each test individually before changing anything"], E14)

M2("web-service", "act-early",
   "You have one strong candidate cause and can deploy a fix to staging in a minute. "
   "What next?",
   "Apply the fix to staging and observe",
   ["Review the remaining candidates before applying anything",
    "Read the deployment pipeline to be sure the fix will land",
    "Reproduce the fault twice more to be certain of the pattern"], E14)

# ---------------------------------------------------------------------------
# D15  STOP WHEN THE CHECK PASSES
# Both groups ended in a verified fix. The expensive group kept acting: its median
# run continued for 144 actions where the cheap group's finished at 22.
# ---------------------------------------------------------------------------
E15 = ("92 runs that all reached the same verified fix: the expensive group continued "
       "to a median of 144 actions against the cheap group's 22, despite the outcome "
       "being identical")

M2("general", "stop-at-done",
   "The check that defines success now passes. You can think of two further "
   "improvements. What is the most economical action?",
   "Stop; the check that defines success already passes",
   ["Make the two improvements while you have the context loaded",
    "Run the check a second time to be certain it really passes",
    "Review the whole change once more before declaring completion"], E15)

M2("build", "stop-at-done",
   "The build succeeds and the artefact is produced. You notice some warnings that were "
   "there before your change. What next?",
   "Stop; the build already succeeds and produces the artefact",
   ["Refactor the code you touched now that it works",
    "Rebuild once more to confirm the success was not a fluke",
    "Read the full build log to be sure nothing was missed"], E15)

M2("testing", "stop-at-done",
   "Every test now passes, including the ones that were failing. What is most "
   "economical?",
   "Stop; every test including the failing ones now passes",
   ["Add tests for the case you just fixed",
    "Run the suite again to be sure",
    "Re-read your change to confirm it is the right fix"], E15)

M2("data-pipeline", "stop-at-done",
   "The pipeline completes and the output matches the expected result exactly. What "
   "next?",
   "Stop; the output already matches the expectation",
   ["Tidy the transform you edited while you are in it",
    "Run the pipeline again to confirm the match",
    "Compare the output field by field a second time"], E15)

# ---------------------------------------------------------------------------
# D16  NARROW WITH EVIDENCE, NOT BY SURVEY
# The expensive group opened by listing files 21.7% of the time against 6.5%.
# ---------------------------------------------------------------------------
E16 = ("first action of 92 runs that all reached the verified fix: the expensive group "
       "opened by listing or surveying files 21.7% of the time against the cheap "
       "group's 6.5%")

M2("general", "narrow-with-evidence",
   "You have a failure message that names a specific symbol. You could also survey the "
   "project layout. What is the most economical next action?",
   "Follow the symbol the message named",
   ["Change every symbol with a similar name to be safe",
    "Survey the largest files, since the fault is likely in one of them",
    "Read the project's documentation to learn the conventions"], E16)

M2("web-service", "narrow-with-evidence",
   "A single failing request carries a trace identifier that appears in the logs. What "
   "next?",
   "Follow that identifier through the logs",
   ["Survey the service's module layout first",
    "Read the logs from the start of the hour for context",
    "List the deployed services to see which could be involved"], E16)

M2("data-pipeline", "narrow-with-evidence",
   "The job failed on one named partition out of hundreds. What is most economical?",
   "Inspect the one partition the job named",
   ["Survey the partition layout to understand the scheme",
    "Sample several partitions to see whether others are affected",
    "List every partition written that day"], E16)

M2("mobile", "narrow-with-evidence",
   "A crash report names one class and one line. What next?",
   "Open the class the crash report named, at that line",
   ["Roll back the release that preceded the crash reports",
    "List the classes changed in the last release",
    "Read the crash reports for other devices for comparison"], E16)

# ---------------------------------------------------------------------------
# D17  DROP A DISPROVEN LINE
# ---------------------------------------------------------------------------
E17 = ("the expensive group's 6.7x action cost at an identical outcome is concentrated "
       "in repetition: 49.6% of its post-run actions were another run and 26.3% of its "
       "post-search actions another search, against 13.9% and 15.2% in the cheap group")

M2("general", "drop-disproven",
   "You have tested your leading hypothesis and the result contradicts it. What is the "
   "most economical next action?",
   "Abandon it and test the next hypothesis",
   ["Test it again with a slightly different method in case the test was flawed",
    "Re-read the code that made you believe it, to see what you misread",
    "Refine the hypothesis so it survives the result"], E17)

M2("testing", "drop-disproven",
   "You believed a fixture was at fault. Replacing it changes nothing. What next?",
   "Abandon the fixture theory and look elsewhere",
   ["Replace all the fixtures the failing tests use",
    "Re-read the fixture to understand why you suspected it",
    "Adjust the hypothesis to a subtler fixture problem"], E17)

M2("performance", "drop-disproven",
   "You suspected I/O was the bottleneck. Measurement shows it is 3% of the time. What "
   "is most economical?",
   "Accept that I/O is not the bottleneck and measure elsewhere",
   ["Optimise the I/O layer anyway, since it is measurable",
    "Read the I/O layer to understand why it is so fast",
    "Refine the hypothesis to a specific I/O path"], E17)

M2("web-service", "drop-disproven",
   "You thought a caching layer was serving stale data. Disabling the cache does not "
   "change the response. What next?",
   "Accept the cache is not involved and investigate elsewhere",
   ["Remove the caching layer from the service entirely",
    "Read the cache implementation to see how it could still be involved",
    "Narrow the hypothesis to one cache key"], E17)

# ---------------------------------------------------------------------------
# D18  END ON A CHECK
# ---------------------------------------------------------------------------
E18 = ("measured on 2,642 post-edit transitions across 92 runs that all reached the "
       "verified fix: the cheap group followed an edit with a run or a check 44.9% of "
       "the time against the expensive group's 30.3%")

M2("general", "end-on-a-check",
   "You believe the work is complete. What is the most economical final action?",
   "Run the check that defines success",
   ["Extend the change to the neighbouring cases while you are here",
    "Summarise what you did so the reasoning is recorded",
    "Review the parts of the system you did not touch for side effects"], E18)

M2("build", "end-on-a-check",
   "You have finished the edits you believe fix the build. What final action?",
   "Run the build, which is the check that defines success",
   ["Tidy the changed files before building",
    "Write up the cause and the fix",
    "Inspect unrelated modules for collateral damage"], E18)

M2("data-pipeline", "end-on-a-check",
   "Your changes to the pipeline are complete in your judgement. What next?",
   "Run it and compare the output against the expectation",
   ["Re-read the transforms you changed",
    "Document the change for the next person",
    "Audit the steps you did not modify"], E18)

M2("mobile", "end-on-a-check",
   "You have made the changes you think fix the crash. What final action?",
   "Launch the app on the affected device",
   ["Re-read the code you changed",
    "Write up the crash cause",
    "Review other screens for the same pattern"], E18)

# ---------------------------------------------------------------------------
# D19  A SEARCH SHOULD LEAD TO A CHANGE
# ---------------------------------------------------------------------------
E19 = ("measured on 237 post-search transitions across 92 runs that all reached the "
       "verified fix: the cheap group moved from a search to an edit 28.8% of the time "
       "against the expensive group's 15.8%")

M2("general", "search-should-produce-changes",
   "Your search located the definition responsible for the defect. What is the most "
   "economical next action?",
   "Change the definition your search just located",
   ["Change every caller as well, in case they share the fault",
    "Read the file it lives in from the top",
    "Search for similar definitions to see whether they share the fault"], E19)

M2("testing", "search-should-produce-changes",
   "A search found the single assertion helper that all the failing tests use, and it is "
   "wrong. What next?",
   "Fix the assertion helper the failing tests share",
   ["Rewrite the failing tests so they no longer use the helper",
    "Read the helper's file completely before editing",
    "Search for other helpers with the same problem"], E19)

M2("web-service", "search-should-produce-changes",
   "A search identified the middleware that strips the header. That is the defect. What "
   "is most economical?",
   "Change the middleware that strips the header",
   ["Move the header handling into a new middleware of your own",
    "Read the middleware chain from the start",
    "Search the client code for where the header is set"], E19)

M2("performance", "search-should-produce-changes",
   "A search located the N+1 query that profiling implicated. What next?",
   "Fix the N+1 query that profiling implicated",
   ["Add a cache in front of the query instead of fixing it",
    "Read the whole data-access module first",
    "Search for the ORM's documentation on eager loading"], E19)

# One extra to reach 77: the composite lesson.
M2("general", "act-not-narrate",
   "You have identified the cause and know the change to make. What is the most "
   "economical next action?",
   "Make the change you have already identified",
   ["Apply the fix and two related improvements together",
    "List the alternatives you considered and why you rejected them",
    "Describe the change in detail so it can be reviewed first"],
   "92 runs that all reached the same verified fix: the expensive group spent 6.7x the "
   "actions (median 144 vs 22) reaching an identical outcome")
