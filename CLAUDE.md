# ABSOLUTELY CRITICAL — DO NOT MAKE MAJOR DESIGN CHOICES ON YOUR OWN

**You are FORBIDDEN from making any major design choice without first consulting the user.** If the user has not explicitly specified an aspect of the design — a data shape, an encoding scheme, a probability semantic, an algorithm, a discretization rule, a loss formulation, a label policy, an architectural decision, a config field's meaning, anything load-bearing for behavior — STOP and ask before choosing.

This rule supersedes auto-mode. "Minimize interruptions, prefer action over planning" does NOT apply when a design choice is unspecified. In that case, asking IS the action.

What counts as a major design choice (non-exhaustive):
- Whether a tensor is binary vs soft probabilities, or sums to 1 vs is multi-label.
- Whether something is computed from ground truth vs from prediction, or shared between input and target.
- Discretization rules (argmax vs threshold vs none), label hierarchies, mutually-exclusive vs independent encodings.
- Loss type (CE vs BCE vs MSE), what constitutes a positive/negative example, what gets masked from the loss.
- Algorithm choices when multiple reasonable options exist (e.g., centroid + cross product vs SVD for ring normals; CB-direction vs SASA for solvent-facing).
- Default hyperparameters that meaningfully bias training behavior (not obvious sizing like a batch dim).
- New noise/augmentation rules, new feature semantics, new conventions in shared code.

When in doubt, ask. A 30-second clarification beats a multi-turn revert.

If the user previously specified one design and you are tempted to "fix" or "improve" it because something looks suboptimal — you are NOT permitted to silently change it. Surface the concern, ask, then act.

---

Make sure any code you write does not have "AI code smell". It should be clean, concise, and human readable. It should be able to make reasonable assumptions, rather than methods hedging for every possible input even if it wouldn't realistically come up.

Avoid these:
Over-defensive code                                                                                                                                                              
  - Try/except around operations that can't realistically fail
  - Null checks on values guaranteed by the type system or prior validation                                                                                                        
  - Validating inputs at every internal function instead of at system boundaries                                                                                                 
  - Fallback branches for "just in case" scenarios that never occur                                                                                                                
                                                                                                                                                                                   
  Over-explanation                                                                                                                                                                 
  - Comments restating what the code already says (# increment counter above i += 1)                                                                                               
  - Docstrings on every trivial helper                                                                                                                                             
  - Comments referencing the task/PR ("added for the login flow", "fixes bug #42")                                                                                               
  - Multi-paragraph block comments explaining obvious logic                                                                                                                        
                                                                                                                                                                                   
  Premature abstraction                                                                                                                                                            
  - Extracting a helper used exactly once                                                                                                                                          
  - Generic utils.py / helpers.py grab bags                                                                                                                                        
  - Config objects, factories, or strategy patterns for two simple cases                                                                                                         
  - Parameters for hypothetical future needs ("in case we want to support X later")                                                                                                
                                                                                                                                                                                   
  Unnecessary churn                                                                                                                                                                
  - Bundling drive-by refactors into a bug fix                                                                                                                                     
  - Renaming/reformatting nearby code the task didn't touch                                                                                                                        
  - Leaving # removed / # deprecated breadcrumbs instead of deleting                                                                                                               
  - Keeping backwards-compat shims when nothing calls the old path                                                                                                                 
                                                                                                                                                                                   
  Hedging                                                                                                                                                                          
  Premature abstraction
  - Extracting a helper used exactly once
  - Generic utils.py / helpers.py grab bags
  - Comments referencing the task/PR ("added for the login flow", "fixes bug #42")
  - Docstrings on every trivial helper
  - Comments referencing the task/PR ("added for the login flow", "fixes bug #42")
  Over-explanation
  - Comments restating what the code already says (# increment counter above i += 1)
  - Docstrings on every trivial helper
  - Comments referencing the task/PR ("added for the login flow", "fixes bug #42")
  - Multi-paragraph block comments explaining obvious logic

  Premature abstraction
  - Extracting a helper used exactly once
  - Generic utils.py / helpers.py grab bags
  - Config objects, factories, or strategy patterns for two simple cases
  - Parameters for hypothetical future needs ("in case we want to support X later")

  Unnecessary churn
  - Bundling drive-by refactors into a bug fix
  - Renaming/reformatting nearby code the task didn't touch
  - Leaving # removed / # deprecated breadcrumbs instead of deleting
  - Keeping backwards-compat shims when nothing calls the old path

  Hedging
  - Returning Optional[T] when the function always returns T
  - isinstance checks before operations the type already guarantees
  - Wrapping single expressions in a function "for clarity"
  - if x is not None and len(x) > 0 and x[0]... chains where one check suffices

  Verbose naming / ceremony
  - get_user_by_id_from_database(user_id) instead of get_user(id)
  - Wrapping stdlib calls in thin pass-through functions
  - Classes with one method that should've been a function
  - Enums/constants for values used exactly once

  Half-finished work
  - TODO comments left in committed code
  - Stub branches that raise NotImplementedError
  - Dead code paths the author "might need later"

  Test smells
  - Mocking the thing under test
  - IMPORTANT: AVOID USING MOCK APPROACHES LIKE MONKEYPATCH
  - Asserting on implementation details instead of behavior                                                                                                                        
  - One giant test covering ten cases instead of focused tests
  - Snapshot tests where a plain equality assertion would be clearer  