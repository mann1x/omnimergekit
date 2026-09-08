#!/usr/bin/env python3
"""Build the GEPO EFFICIENCY pool: generic troubleshooting/development policy.

WHERE THIS COMES FROM
---------------------
Four arms of the manic_miner harness, 27 runs with recoverable event logs, measured
2026-09-08. One arm reached the same verified outcome roughly an order of magnitude
cheaper than the others, and the gap is not subtle:

    arm                n   iters  tools   out_tok  tok/action  rerun  reread  edit->verify
    qwen3.6-base       5      28     26    26,424       1,052      0       1          0.50
    ornith-27b-coder   4      46     58   150,953       1,734      3       1          0.49
    a3b-coder         15     116    117   194,630       2,369      8       9          0.27
    v7-coder           3      77     74   234,078       4,174      2       4          0.25

    rerun  = consecutive identical program runs with no edit between them
    reread = consecutive reads with nothing changed between them
    edit->verify = fraction of edits immediately followed by a check

The efficient arm is not smarter per action; it takes FEWER actions and says less per
action. Its runs all open by running the program. The expensive arms open by reading
around it, re-run without having changed anything, re-read what they already hold, and
batch edits before checking any of them.

WHAT THIS POOL IS, AND WHAT IT IS NOT
-------------------------------------
The items below are decision points, abstracted away from the task those runs happened
to be doing. Nothing here mentions HTML, games, a particular agent, or a particular
tool: every scenario is stated in plain prose, the options are plain prose, and the
answer is a letter. A parser renders them for whatever target model is being trained,
in whatever thinking format and tool dialect that model uses. Nothing in this file
commits to one.

Be clear about the limit: this teaches a POLICY over described situations. It is not an
agentic pool, and picking the cheap option in a described situation is not the same
skill as being cheap inside a real loop. What does transfer directly is the second half
of the lesson -- `length_lambda > 0` puts every correct rollout of a group into the
length-shaped band, so reaching the right decision in fewer tokens outscores reaching it
in more. That is brevity pressure on troubleshooting reasoning specifically, which is
what the measurement above says the expensive arms lack.

ONE IMPLEMENTATION FACT WORTH KNOWING
-------------------------------------
In gepo_reward_v2, `length_lambda` is a GATE, not a magnitude:

    if P and max(lam_f[i] for i in P) <= 0.0:   -> correctness-only, reward 1.0
    otherwise                                    -> base + alpha * s_i (length-shaped)

The number above zero never multiplies anything. 0.7 and 0.05 behave identically. This
pool therefore sets 1.0 for "length-shaped" and 0.0 for "correctness-only" and does not
pretend the value carries more meaning than it does.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import sys

PROMPT_TMPL = (
    "What is the correct answer to this question: {q}\n\n"
    "Choices:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n\n"
    'Give a step-by-step reasoned answer, then finish with "The correct answer is (X)" '
    "where X is A, B, C or D."
)

# Each item: (domain, principle, question, correct, [three wrong])
# The wrong options are the measured failure modes, not strawmen: surveying before
# reproducing, repeating an action that cannot return new information, widening the
# read, batching changes before any check, and narrating instead of acting.
ITEMS: list[tuple[str, str, str, str, list[str]]] = []


def S(domain, principle, q, correct, wrong, evidence=None):
    ITEMS.append((domain, principle, q, correct, wrong, evidence))


# ---------------------------------------------------------------------------
# MINED DECISION POINTS
#
# The block above is authored from the aggregate metrics. This block is different:
# each item below is a point where the two groups' conditional action distributions
# actually diverge, measured over 109 runs by counting what each group did next from
# the same preceding action. The evidence string on every row carries the observed
# split and the sample size, so a reviewer can check the claim rather than take it.
#
# Efficient group  = qwen3.6-base + omnimerge-v4  (75 runs)
# Inefficient group = a3b-coder, v7-coder, ornith-27b-coder, omnimerge-v6 (33 runs)
#
# omnimerge-v4 belongs in the efficient group on ACTIONS and is the reason the
# identity confound is now controlled: it shares the qwen3.6 base with the efficient
# arm and matches it action for action (28 turns, 28 calls, 8 edits, 4 runs, 2 checks,
# 7 reads at the median) while spending 1.22x the tokens per turn. Same base, same
# policy, more words -- which is why the mined items below are about ACTION CHOICE and
# the length objective is left to carry verbosity separately. No decision item can
# teach what separates base from v4, because on actions nothing does.
# ---------------------------------------------------------------------------
def M(domain, principle, q, correct, wrong, evidence):
    S(domain, principle, q, correct, wrong, evidence)


# ---------------------------------------------------------------------------
# P1  REPRODUCE BEFORE SURVEYING
# Every efficient run opened by running the thing. The expensive arms opened by
# reading around it, and paid for a map they did not yet know how to read.
# ---------------------------------------------------------------------------
S("testing", "reproduce-first",
  "A colleague reports that a test suite fails on the main branch. You have the "
  "repository checked out and have not run anything yet. What is the most economical "
  "first action?",
  "Run the test suite and read the failure output",
  ["Read the test files most likely to be involved, to build a picture first",
   "Read the last twenty commits to find what probably broke it",
   "Ask the colleague which test they think is at fault"])

S("web-service", "reproduce-first",
  "A service is returning HTTP 500 for some requests in production. You have access to "
  "logs, metrics, and the code. Nothing has been inspected yet. What is the most "
  "economical first action?",
  "Find one failing request in the logs and read its error and stack trace",
  ["Read the request-handling module from the top to understand the flow",
   "Review the deployment history for the last month",
   "Add more logging around every handler and wait for the next failure"])

S("build", "reproduce-first",
  "A project no longer builds on a teammate's machine but builds on yours. What is the "
  "most economical first action?",
  "Get the exact build command and full error text from their failing run",
  ["Compare your installed toolchain versions against theirs, field by field",
   "Read the build configuration files looking for anything environment-dependent",
   "Have them delete all build artefacts and dependencies and start over"])

S("data-pipeline", "reproduce-first",
  "A nightly job produced a report with numbers that are clearly wrong. What is the "
  "most economical first action?",
  "Re-run the job on the same input and confirm it reproduces the wrong numbers",
  ["Read the transformation code looking for a logic error",
   "Check whether any upstream schema changed recently",
   "Compare the last thirty days of reports to find when the drift began"])

S("performance", "reproduce-first",
  "Users report that one page of an application became slow this week. What is the most "
  "economical first action?",
  "Measure the slow page directly and see where the time is actually spent",
  ["Read the code paths that page uses, looking for obviously costly operations",
   "Review this week's merged changes for anything performance-related",
   "Add caching to the queries that page issues and see whether users stop complaining"])

S("mobile", "reproduce-first",
  "An application crashes on launch for some users but not for you. What is the most "
  "economical first action?",
  "Obtain one crash report and read the exception and stack it captured",
  ["Read the launch sequence code and reason about what could fail",
   "Ask the affected users what device and version they are on and study the matrix",
   "Wrap the launch path in error handling so it no longer crashes"])

# ---------------------------------------------------------------------------
# P2  DO NOT REPEAT AN ACTION THAT CANNOT RETURN NEW INFORMATION
# Measured: 8 consecutive re-runs with no intervening edit in the median a3b run,
# 0 in the efficient arm.
# ---------------------------------------------------------------------------
S("testing", "no-null-actions",
  "You ran a failing test and read its output. You have not changed anything since. "
  "What is the least useful thing you could do next?",
  "Run the same test again unchanged",
  ["Read the function named in the top stack frame",
   "Write down what the error says the state was at the point of failure",
   "Check whether the test passes on an earlier commit"])

S("web-service", "no-null-actions",
  "A deployment failed and you have read the full deployment log. Nothing about the "
  "system or the configuration has changed since. What is the least useful next action?",
  "Trigger the same deployment again to see whether it fails the same way",
  ["Read the configuration value the log says was rejected",
   "Compare the failing configuration against the last successful one",
   "Check whether the target environment is reachable at all"])

S("data-pipeline", "no-null-actions",
  "A query returned an empty result set. You have confirmed the query is syntactically "
  "valid and ran it twice with identical results. What should you do next?",
  "Test whether the filter conditions match anything by relaxing them one at a time",
  ["Run the query a third time in case the result is intermittent",
   "Rewrite the query in a different style and run that instead",
   "Read the full schema of every table the query touches"])

S("build", "no-null-actions",
  "A compilation error names a specific file and line. You have read that line. What is "
  "the least useful next action?",
  "Recompile without changing anything, to see the error again",
  ["Read the declaration of the symbol that line references",
   "Check whether that file changed recently",
   "Look for other uses of the same symbol that do compile"])

# ---------------------------------------------------------------------------
# P3  DO NOT RE-ACQUIRE WHAT YOU ALREADY HAVE
# Measured: 9 consecutive repeated reads in the median a3b run, 1 in the efficient arm.
# ---------------------------------------------------------------------------
S("general", "no-reacquisition",
  "You read a configuration file forty seconds ago and it has not changed. You need one "
  "value from it that you already saw. What should you do?",
  "Use the value you already read",
  ["Read the file again to be certain the value is current",
   "Read the file again and also read the files it references",
   "Search the codebase for other places that value might be defined"])

S("testing", "no-reacquisition",
  "You have already read a function in full and understood it. A later step needs to "
  "know its parameter order. What is the most economical action?",
  "Recall the parameter order from what you already read",
  ["Read the whole function again to be sure",
   "Read the function and every function that calls it",
   "Search the repository for all invocations to infer the order"])

S("web-service", "no-reacquisition",
  "You are debugging and have read the same log excerpt three times without changing "
  "anything in between. What does that pattern most likely indicate?",
  "You are re-reading instead of acting on what the log already told you",
  ["The log is unreliable and should be read once more carefully",
   "You need to enable a more verbose log level",
   "The problem is not visible in logs and needs a debugger"])

# ---------------------------------------------------------------------------
# P4  VERIFY EACH CHANGE IMMEDIATELY
# Measured: half of the efficient arm's edits are checked on the next action; a quarter
# of the expensive arms' are. Batched edits make a failure ambiguous across all of them.
# ---------------------------------------------------------------------------
S("general", "verify-each-change",
  "You have identified four separate changes you believe will fix a defect. What is the "
  "most economical way to apply them?",
  "Apply one, verify, and only then apply the next",
  ["Apply all four, then verify once, since verification is the slow part",
   "Apply all four and verify each one afterwards in order",
   "Apply the two you are most confident about, then verify"])

S("testing", "verify-each-change",
  "You applied six changes at once and the test suite still fails, with a different "
  "error than before. What is the cheapest way to regain footing?",
  "Revert to the last verified state and reapply one change at a time",
  ["Read the new error and apply a seventh change to address it",
   "Revert only the change you now believe was wrong and re-run",
   "Add logging to all six changed regions and run again"])

S("build", "verify-each-change",
  "A change you made is intended to fix a build error. What should happen immediately "
  "after you make it?",
  "Build again and read the result",
  ["Make the next planned change, then build once at the end",
   "Read the changed file again to confirm the edit landed correctly",
   "Commit the change so it is not lost, then continue"])

S("web-service", "verify-each-change",
  "You are about to report that a defect is fixed. Your last action was an edit. What "
  "must you do first?",
  "Run the check that reproduces the defect and confirm it now passes",
  ["Re-read the edited region to confirm the change is correct",
   "Summarise the changes you made and why they should work",
   "Check that no other tests were affected by reading the surrounding code"])

# ---------------------------------------------------------------------------
# P5  LET THE ERROR NARROW THE SEARCH, DO NOT WIDEN THE READ
# Measured: the expensive arms read 1.12 files per edit, the efficient arm 0.71.
# ---------------------------------------------------------------------------
S("testing", "narrow-with-evidence",
  "A stack trace names a function, a file, and a line. What should you read first?",
  "That line and the function containing it",
  ["The whole file, so the function is understood in context",
   "The module's public interface, to understand the design",
   "Every caller of that function, to see which one passed bad input"])

S("data-pipeline", "narrow-with-evidence",
  "A job fails with an error naming one specific column as having an unexpected type. "
  "What is the most economical next step?",
  "Inspect the values actually present in that column",
  ["Read the full schema definition for the table",
   "Read the transformation code for all columns to find similar issues",
   "Compare the current schema against the one from last month"])

S("web-service", "narrow-with-evidence",
  "An error message says a required configuration key is missing. What should you do "
  "first?",
  "Check whether that key is set in the environment the failure came from",
  ["Read the configuration loading code to understand how keys are resolved",
   "Compare the configuration across all environments",
   "Add a default value for the key so the error cannot occur"])

S("performance", "narrow-with-evidence",
  "A profiler shows one function accounting for 80% of runtime. What should you examine?",
  "That function and what it is doing with the time",
  ["The call graph in full, to understand overall structure",
   "The next four functions in the profile as well, for context",
   "The algorithmic complexity of the whole subsystem"])

S("general", "narrow-with-evidence",
  "You have two competing hypotheses about a defect. One observation would rule out "
  "exactly one of them; another observation is more thorough but takes ten times as "
  "long and rules out neither cleanly. What should you do?",
  "Make the cheap observation that discriminates between them",
  ["Make the thorough observation, since more information is better",
   "Make both, starting with the thorough one",
   "Pick the more likely hypothesis and act on it without observing"])

# ---------------------------------------------------------------------------
# P6  ABANDON A DISPROVEN HYPOTHESIS
# ---------------------------------------------------------------------------
S("general", "drop-disproven",
  "You believed a defect was caused by a particular function. You read it and it "
  "provably cannot produce the observed behaviour. What should you do?",
  "Discard that hypothesis and look for evidence pointing elsewhere",
  ["Read the function once more in case something was missed",
   "Change the function anyway, since it could be improved",
   "Read every function it calls, since the cause is probably nearby"])

S("web-service", "drop-disproven",
  "You suspected a caching layer was serving stale data. You confirmed the cache was "
  "disabled during the failure. What should you conclude?",
  "The cache is not the cause and the search should move elsewhere",
  ["The cache may still be involved through an indirect path worth tracing",
   "The cache configuration should be reviewed anyway while you are here",
   "The test was inconclusive and should be repeated"])

S("testing", "drop-disproven",
  "You have made the same edit three times in slightly different forms and the check "
  "fails identically each time. What does this most strongly suggest?",
  "The thing you are changing is not what is causing the failure",
  ["The edit is correct but needs to be combined with another change",
   "The check is unreliable and should be replaced",
   "A fourth variation of the edit is likely to succeed"])

# ---------------------------------------------------------------------------
# P7  STOP WHEN THE CRITERION IS MET
# ---------------------------------------------------------------------------
S("general", "stop-at-done",
  "The check that defines the task as complete now passes. You can see three unrelated "
  "things in the code you would like to improve. What should you do?",
  "Stop; the task is done",
  ["Make the three improvements while you have the context loaded",
   "Make the smallest of the three, since it is nearly free",
   "Re-run the check several more times to be confident it is stable"])

S("testing", "stop-at-done",
  "A defect is fixed and verified. What is the appropriate amount of additional "
  "refactoring to perform as part of this task?",
  "None, unless it was asked for",
  ["Whatever can be done safely in the same area",
   "Enough to prevent the same class of defect recurring",
   "Any cleanup that does not change behaviour"])

# ---------------------------------------------------------------------------
# P8  ACT RATHER THAN NARRATE
# Measured as tokens per action: 1,052 for the efficient arm against 4,174 for the most
# expensive one, for the same class of outcome.
# ---------------------------------------------------------------------------
S("general", "act-not-narrate",
  "You have decided exactly which change to make and why. What should you do next?",
  "Make the change",
  ["Write out the full plan and rationale before making it",
   "Restate the reasoning once more to check it is sound",
   "List the alternatives you considered and why you rejected them"])

S("general", "act-not-narrate",
  "You have spent a long stretch analysing a problem without taking any action, and the "
  "analysis is no longer producing new conclusions. What is the best response?",
  "Take the smallest concrete action that would produce new information",
  ["Continue analysing until the answer is certain",
   "Restart the analysis from the beginning with fresh assumptions",
   "Write a summary of the analysis so far and continue from it"])

S("general", "act-not-narrate",
  "Two approaches would both plausibly resolve a problem. One can be tried and checked "
  "in a minute; the other would take an hour to evaluate on paper. What should you do?",
  "Try the one-minute approach and see what happens",
  ["Evaluate both on paper before committing to either",
   "Reason carefully about which is more likely correct, then do that one",
   "Try both simultaneously to save time"])

S("general", "act-not-narrate",
  "You are uncertain which of four possible causes is responsible, and a single cheap "
  "experiment would eliminate two of them. What should you do?",
  "Run the experiment",
  ["Rank the four causes by likelihood and investigate the most likely",
   "Research each cause thoroughly before experimenting",
   "Address all four causes at once so the problem is resolved regardless"])


# --- mined: time to the first change ----------------------------------------
TTFC = ("measured across 109 runs: the efficient group made its first change after a "
        "median of 3 actions, the inefficient group after 8")

M("general", "act-early",
  "You have run the failing command, read its error, and looked at the one function it "
  "names. You have a plausible fix in mind but are not certain. What should you do?",
  "Make the change and see what the check says",
  ["Read two or three more related files to raise your confidence first",
   "Look for similar patterns elsewhere in the codebase to confirm the diagnosis",
   "Trace the full call path that reaches the function before touching it"], TTFC)

M("general", "act-early",
  "Eight actions into a task you have run the program twice, read five files, and "
  "changed nothing. What is the most likely explanation?",
  "You are delaying the first change longer than the evidence requires",
  ["The problem is genuinely hard and needs more investigation",
   "You are being appropriately rigorous before modifying code",
   "You have not yet found the file that contains the defect"], TTFC)

M("testing", "act-early",
  "Which sequence is more economical for a defect with a clear error message?",
  "Run, read the named function, change it, check",
  ["Run, read the file, read its imports, read the tests, change it, check",
   "Read the tests, read the file, run, read the output again, change it, check",
   "Run, run again to confirm, read the file, read its callers, change it, check"],
  TTFC)

# --- mined: how a run should end --------------------------------------------
ENDING = ("measured on 109 runs: 60% of efficient runs ended on a verification step, "
          "against 35% of inefficient runs, which more often ended on a bare program "
          "run or on nothing at all")

M("general", "end-on-a-check",
  "What should the final action of a completed debugging task be?",
  "A verification that the original problem is gone",
  ["A summary of the changes made and why",
   "A final read of the changed files to confirm they look right",
   "A broader test of the surrounding functionality"], ENDING)

M("general", "end-on-a-check",
  "You have made your last edit and believe the task is complete. You have not run "
  "anything since that edit. What is the state of the task?",
  "Unverified; the last change has not been checked",
  ["Complete, since the change was small and clearly correct",
   "Complete, since earlier checks passed",
   "Complete, but worth a final review of the code"], ENDING)

# --- mined: streaks ---------------------------------------------------------
STREAK = ("measured across 109 runs: the inefficient group produced 64 streaks of three "
          "or more identical consecutive actions against the efficient group's 20, with "
          "one streak reaching 33 consecutive runs of the same command")

M("general", "break-the-streak",
  "You have run the same command five times in a row, changing nothing between the "
  "runs. What should you do?",
  "Stop running it and change something, or change what you are asking it",
  ["Run it a sixth time with more verbose output enabled",
   "Run it once more to be certain the behaviour is stable",
   "Run it in a different environment to compare"], STREAK)

M("general", "break-the-streak",
  "You notice you have read four files in a row without making any change. What is the "
  "most useful interpretation?",
  "The reading is no longer informing a decision and should stop",
  ["You are building necessary context and should continue",
   "You should read the remaining related files for completeness",
   "You should start again and read them more carefully"], STREAK)

M("data-pipeline", "break-the-streak",
  "A long sequence of identical actions with no change between them is best described "
  "as what?",
  "Repetition that cannot produce new information",
  ["Thoroughness that reduces the chance of a mistake",
   "A reasonable way to rule out intermittent behaviour",
   "A sign that the tooling is unreliable"], STREAK)

# --- mined: late-phase behaviour --------------------------------------------
LATE = ("measured by phase across 109 runs: in the last third of a run the efficient "
        "group spent 15% of its actions reading and 13% verifying, while the "
        "inefficient group spent 26% reading and 7% verifying")

M("general", "converge-late",
  "You are late in a task, close to a fix. What should the balance of your remaining "
  "actions be?",
  "Mostly verifying, with reading only when a check points somewhere specific",
  ["Mostly reading, to be certain the fix is correct before finishing",
   "An even split between reading and verifying",
   "Mostly running the program, to build confidence through repetition"], LATE)

M("general", "converge-late",
  "Late in a task you find yourself reading as much as you were at the start. What does "
  "that suggest?",
  "You have not converged and are still searching rather than closing",
  ["You are doing final due diligence, which is appropriate",
   "The task is larger than it first appeared",
   "You should read faster to make up the time"], LATE)


# Measured 2026-09-08 across 27 harness runs with recoverable event logs. These are the
# numbers a length budget should be set from, rather than a guess:
#
#   output tokens per assistant turn, median per arm
#     qwen3.6-base       1052   (p25 921, p75 1468)   <- the efficient arm
#     ornith-27b-coder   1729
#     a3b-coder          2336
#     v7-coder           4046
#
#   ratio efficient:expensive  = 0.61, 0.45, 0.26  -> median 0.45
#
# For reference, the AN E2B run that worked set its budget at ~0.66 of the observed
# mean. 0.45 is a harder pull than the recipe known to work here, so a first epoch
# should use the milder end and the ratio should be applied to the TARGET model's own
# measured median, not to 1052 directly -- 1052 is tokens per agent turn, which is a
# different quantity from tokens per completion on this pool.
EFFICIENT_TOKENS_PER_TURN = 1052
CROSS_FAMILY_RATIO = 0.45   # base vs the coder arms -- CONFOUNDED by model identity
CONTROLLED_RATIO = 0.82     # base vs omnimerge-v4: same base, same action counts
CONSERVATIVE_RATIO = 0.66   # the ratio the AN E2B run that worked actually used



# --- mined: the opening move ------------------------------------------------
# efficient 51% run / 27% read / 21% list ; inefficient 30% run / 64% read
# n = 75 efficient runs, 33 inefficient runs
OPENING = ("measured over 109 runs: the efficient group opened by running the program "
           "in 51% of runs and by reading in 27%; the inefficient group opened by "
           "reading in 64% and by running in 30%")

M("general", "reproduce-first",
  "You have been handed a defect report for a system you have not seen before, and a "
  "command that is said to demonstrate the problem. What should your first action be?",
  "Run the command and read what it produces",
  ["Read the source files most likely to be involved",
   "Read the project documentation to understand the architecture",
   "List the repository contents to get oriented"], OPENING)

M("web-service", "reproduce-first",
  "You are given a failing system and both a way to observe the failure directly and a "
  "codebase to read. Which order costs less?",
  "Observe the failure first, then read only what the observation implicates",
  ["Read the codebase first, then observe the failure with that context in hand",
   "Read and observe in parallel so neither blocks the other",
   "Read the codebase thoroughly; observation is unnecessary if the code is understood"],
  OPENING)

# --- mined: what follows a verification -------------------------------------
# after check_file: efficient reads 10%, inefficient reads 40% (-30 pts)
# n = 110 efficient transitions, 177 inefficient
AFTER_CHECK = ("measured on 287 transitions following a verification: the efficient "
               "group went on to read a file 10% of the time, the inefficient group 40%")

M("general", "act-on-the-result",
  "You ran a check and it reported a specific problem. What should you do next?",
  "Act on what the check reported",
  ["Read the surrounding code to understand the context before acting",
   "Read the file the check named, in full, from the beginning",
   "Run a second, broader check to confirm the first"], AFTER_CHECK)

M("testing", "act-on-the-result",
  "A verification step just told you precisely which assertion failed and with what "
  "values. What is the least economical response?",
  "Begin reading files to build background understanding",
  ["Change the code path that produced the wrong value",
   "Look at the one function that computes the value in the assertion",
   "Re-run the verification after making a change"], AFTER_CHECK)

# --- mined: what follows a read ---------------------------------------------
# after read_files: efficient edits 58% vs 37%; inefficient runs 25% vs 11%
# n = 576 efficient transitions, 887 inefficient
AFTER_READ = ("measured on 1,463 transitions following a read: the efficient group made "
              "a change next 58% of the time against the inefficient group's 37%, which "
              "instead ran the program again 25% of the time against 11%")

M("general", "reads-should-produce-changes",
  "You have just read the code you believed was responsible for a defect, and it "
  "confirms your hypothesis. What should you do next?",
  "Make the change it implies",
  ["Run the program again to confirm the failure is still present",
   "Read the functions it calls, to be thorough before changing anything",
   "Read the tests covering it to understand the expected behaviour"], AFTER_READ)

M("general", "reads-should-produce-changes",
  "Over the last several steps you have read six files and changed nothing. What does "
  "this most likely indicate?",
  "You are gathering information you are not using, and should act on what you have",
  ["You are being appropriately careful before making a risky change",
   "You need to read more broadly to find the real cause",
   "The problem is more complex than expected and needs a fresh approach"],
  AFTER_READ)

# --- mined: what follows running the program --------------------------------
# after run_commands: efficient runs again 21%, inefficient 38% (-17 pts)
# n = 478 efficient transitions, 869 inefficient
AFTER_RUN = ("measured on 1,347 transitions following a program run: the inefficient "
             "group's next action was to run it again 38% of the time, against the "
             "efficient group's 21%")

M("general", "no-null-actions",
  "You ran the program and read its output. Between then and now you have changed "
  "nothing. What does running it again get you?",
  "Nothing; the output cannot differ",
  ["Confirmation that the failure is reproducible rather than intermittent",
   "A second chance to notice something missed in the first output",
   "A cleaner output to work from now that the environment is warm"], AFTER_RUN)

M("data-pipeline", "no-null-actions",
  "A job is genuinely intermittent: it fails roughly one run in five. You have run it "
  "once and it failed. What is the most economical next step?",
  "Use the failure you already captured, since you have the evidence you needed",
  ["Run it four more times to establish the failure rate precisely",
   "Run it again to see whether the same failure recurs",
   "Run it repeatedly until it passes, to compare a good run against a bad one"],
  AFTER_RUN)

# --- mined: what follows a search -------------------------------------------
# after search_codebase: efficient reads the hit 53% vs 33%; inefficient runs 18% vs 1%
# n = 116 efficient transitions, 95 inefficient
AFTER_SEARCH = ("measured on 211 transitions following a search: the efficient group "
                "read one of the results next 53% of the time against 33%, while the "
                "inefficient group jumped to running the program 18% of the time "
                "against the efficient group's 1%")

M("general", "follow-the-search",
  "You searched the codebase for a symbol and got a short list of locations. What "
  "should you do next?",
  "Read the location that the search says is most likely relevant",
  ["Run the program to see the symbol's effect at runtime",
   "Search again with a broader term to make sure nothing was missed",
   "Read every location in the list before deciding"], AFTER_SEARCH)

M("general", "follow-the-search",
  "A search returns one exact match and eleven partial ones. What is the most "
  "economical next action?",
  "Read the exact match",
  ["Read the exact match and the three most similar partial ones",
   "Refine the search until only exact matches remain",
   "Run the program to determine which of the twelve is actually reached"],
  AFTER_SEARCH)


def build(seed: int, length_lambda: float, budget: int | None,
          think: bool = True) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for domain, principle, q, correct, wrong, evidence in ITEMS:
        opts = [correct] + list(wrong)
        # Shuffle so the gold is not positionally predictable, and so option ORDER
        # carries no signal. Checked after the fact by audit(); a pool whose gold is
        # guessable from shape teaches the shape, not the policy.
        rng.shuffle(opts)
        gold = "ABCD"[opts.index(correct)]
        prompt = PROMPT_TMPL.format(q=q, a=opts[0], b=opts[1], c=opts[2], d=opts[3])
        h = hashlib.sha256(q.encode()).hexdigest()
        meta = {
            "reward_kind": "mc_letter",
            # GATE semantics in gepo_reward_v2: <=0 means correctness-only, anything
            # above 0 puts the group in the length-shaped band. The magnitude is inert
            # there, so this is deliberately 1.0/0.0 and not a fake-precise number.
            "length_lambda": float(length_lambda),
            "think": think,
            "choices": opts,
            "domain": domain,
            "subdomain": principle,
            "question_sha256": h,
            # Carried whether or not it is used, so an absolute-budget reward can read
            # it without the pool being rebuilt. See the block above for provenance.
            "length_budget": budget,
            "length_budget_provenance": (
                f"efficient arm {EFFICIENT_TOKENS_PER_TURN} tok/turn; "
                f"identity-controlled ratio {CONTROLLED_RATIO} (base vs omnimerge-v4)"),
            # Present only on mined rows: the measured split this item encodes.
            **({"evidence": evidence} if evidence else {}),
            "tier": "mined" if evidence else "authored",
        }
        rows.append({"id": f"eff/{principle}/{h[:12]}", "source": "manic-arm-contrast",
                     "prompt": prompt, "gold": gold, "meta": meta})
    rng.shuffle(rows)
    return rows


def audit(rows: list[dict]) -> int:
    """Refuse a pool whose answer is guessable without reading the question.

    Three ways an MC pool leaks: the gold sits in one position too often, the gold is
    systematically the longest or shortest option, or one principle dominates so the
    policy learnt is a single rule. All three are checked, and the first two are hard
    failures -- a pool that teaches "always pick the short one" has taught nothing
    about troubleshooting.
    """
    bad = 0
    pos = collections.Counter(r["gold"] for r in rows)
    n = len(rows)
    print("gold position:", dict(sorted(pos.items())))
    for letter, count in pos.items():
        if count > 0.40 * n:
            print(f"  FAIL: gold is {letter} in {100*count/n:.0f}% of rows (>40%)")
            bad += 1
    shortest = sum(1 for r in rows
                   if r["meta"]["choices"].index(min(r["meta"]["choices"], key=len))
                   == "ABCD".index(r["gold"]))
    longest = sum(1 for r in rows
                  if r["meta"]["choices"].index(max(r["meta"]["choices"], key=len))
                  == "ABCD".index(r["gold"]))
    print(f"gold is the shortest option in {shortest}/{n} rows ({100*shortest/n:.0f}%), "
          f"the longest in {longest}/{n} ({100*longest/n:.0f}%)")
    if shortest > 0.50 * n:
        print("  FAIL: 'pick the shortest option' would score above chance too often")
        bad += 1
    if longest > 0.50 * n:
        print("  FAIL: 'pick the longest option' would score above chance too often")
        bad += 1
    dom = collections.Counter(r["meta"]["domain"] for r in rows)
    pri = collections.Counter(r["meta"]["subdomain"] for r in rows)
    print("domains:   ", dict(sorted(dom.items())))
    print("principles:", dict(sorted(pri.items())))
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--length-lambda", type=float, default=0.0,
                    help="How hard the length term pulls on this tier, 0.0-1.0. Zero "
                         "(the default) is correctness-only. Now that lambda is a "
                         "MAGNITUDE rather than a gate, a small value is usable here "
                         "where all-or-nothing was not: on a four-way multiple choice "
                         "the cheapest route to a high reward is to answer instantly "
                         "with no reasoning, and at full strength that is the length "
                         "collapse GEPO's asymmetric shaping exists to damp. Something "
                         "like 0.3 applies a real pull while leaving correctness worth "
                         "far more than brevity. Requires the magnitude mode.")
    ap.add_argument("--no-think", action="store_true",
                    help="Write meta.think=False. The tier defaults to thinking ON, "
                         "because the verbosity this pool exists to price lives in the "
                         "reasoning -- a no-think row gives the length term almost "
                         "nothing to shape. The cost is the reason to consider it: "
                         "mc_letter no-think is priced at 1,595 tok in the mixed "
                         "builder, and the thinking cost of THESE items has not been "
                         "measured. Measure the tier before pricing a run on it.")
    ap.add_argument("--length-budget", type=int, default=None,
                    help="Absolute per-completion token budget written to "
                         "meta.length_budget. Set it to the ratio above times the TARGET "
                         "model's measured median completion length on this pool -- "
                         "measure first, do not guess.")
    a = ap.parse_args()
    if not 0.0 <= a.length_lambda <= 1.0:
        sys.exit(f"REFUSE: --length-lambda must be in [0,1], got {a.length_lambda}. "
                 "Above 1 pushes a passing rollout past BASE+ALPHA and breaks the "
                 "invariant that the worst passer still beats the best failure.")
    rows = build(a.seed, a.length_lambda, a.length_budget, not a.no_think)
    bad = audit(rows)
    if bad:
        sys.exit(f"REFUSE: {bad} audit failure(s). A pool that can be answered from the "
                 "shape of the options must not be written.")
    pathlib.Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nwrote {len(rows)} rows -> {a.out}")
    print(f"length_lambda = {a.length_lambda}  "
          f"({'correctness-only' if a.length_lambda <= 0 else 'length-shaped band'})")
    if a.length_lambda > 0:
        print("NOTE: this requires GEPO_LENGTH_LAMBDA_MODE=magnitude (the default). "
              "Under the older 'gate' reading any value above zero applies the FULL "
              "length term, and this tier would pull as hard as 1.0.")


if __name__ == "__main__":
    main()
