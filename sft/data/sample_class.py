"""How a sample is allowed to be trained on -- the one contract every dataset
and the model both speak.

Every dataset emits ``sample_class`` on each sample and ``Motus.training_step``
routes on it. Four on-disk outcome pools collapse onto these four codes, one
branch each:

    * ``MAIN`` -- ``success/``. Eligible for policy, world-model and value.
    * ``TASK_IRRELEVANT`` -- ``fail/``. Tried the task and missed. Never
      policy; value is the late-collapse negative curve.
    * ``SUBOPTIMAL`` -- ``suboptimal/``. Reached the goal by a path not worth
      imitating. Never policy; value is the success curve.
    * ``RANDOM`` -- ``random/``. Never doing the task. Never policy; value is
      the first-frame-steep negative curve.

Every class above ``MAIN`` is off the policy loss, so ``sample_class > 0``
remains the single question "is this off the policy loss".
"""

from enum import IntEnum


class SampleClass(IntEnum):
    MAIN = 0
    TASK_IRRELEVANT = 1
    SUBOPTIMAL = 2
    RANDOM = 3


#: Batch key the classes travel under, so a typo is an ImportError, not a
#: silently absent tensor that turns every sample into ``MAIN``.
SAMPLE_CLASS_KEY = "sample_class"
