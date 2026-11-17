"""How a sample is allowed to be trained on -- the one contract every dataset
and the model both speak.

Every dataset emits ``sample_class`` on each sample and ``Motus.training_step``
routes each loss according to the class:

* ``MAIN`` -- ordinary on-task data. Eligible for the policy, world-model and
  value losses alike.
* ``TASK_IRRELEVANT`` -- the episode is not this task at all. Never trained as
  policy, and its value target is the decaying-slope curve: negative almost
  immediately, because nothing about it was ever on task.
* ``SUBOPTIMAL`` -- the task is being performed, just not well. Never trained as
  policy either, since imitating it teaches the bad version, but it is honest
  video and honest value evidence, so it still trains the world model and the
  value head.

The two non-``MAIN`` classes are both truthy, so ``sample_class > 0`` remains
the single question "is this off the policy loss", and code that only needs
that answer does not have to enumerate the classes.
"""

from enum import IntEnum


class SampleClass(IntEnum):
    MAIN = 0
    TASK_IRRELEVANT = 1
    SUBOPTIMAL = 2


#: Batch key the classes travel under, so a typo is an ImportError, not a
#: silently absent tensor that turns every sample into ``MAIN``.
SAMPLE_CLASS_KEY = "sample_class"
