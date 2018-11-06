from chunk import Chunk
import math
import random

class Model(object):

    # Model parameters

    ga = 1.0 # spreading activation from the goal (:ga; default: 1.0)

    d = 0.5 # decay (:bll; default: 0.5)
    s = 0 # scale of activation noise (:ans; default = 0)

    lf = 1.0 # latency factor (:lf; default: 1.0)
    le = 1.0 # latency exponent (:le; default: 1.0)


    def __init__(self):
        self.time = 0
        self.goal = None
        self.dm = []


    def add_encounter(self, chunk):
        """
        Add an encounter of a specified chunk at the current time.
        If the chunk does not exist yet, create it first.
        """

        # If a chunk by this name does not yet exist, add it to DM
        if chunk.name not in [chunk.name for chunk in self.dm]:
            self.dm.append(chunk)
        
        # If a chunk by this name does exist, ensure that it has the same slots and slot values
        chunk_idx = [i for i, j in enumerate(self.dm) if j.name == chunk.name][0]
        if self.dm[chunk_idx].slots != chunk.slots:
            raise ValueError("Trying to add an encounter to a chunk with the same name (%s) but different slots and/or slot values" % chunk.name)

        # Add an encounter at the current time
        self.dm[chunk_idx].add_encounter(self.time)


    def get_activation(self, chunk):
        """
        Get the activation of the specified chunk at the current time.
        """
        # The specified chunk should exist in DM
        if chunk not in self.dm:
            raise ValueError("The specified chunk (%s) does not exist in DM" % str(chunk.name))

        chunk_idx = [i for i, j in enumerate(self.dm) if j.name == chunk.name][0]
        c = self.dm[chunk_idx]

        # There should be at least one past encounter of the chunk
        if self.time <= min(c.encounters):
            raise ValueError("Chunk %s not encountered at or before time %s" % (str(c.name), str(self.time)))

        baselevel_activation = math.log(sum([(self.time - encounter) ** -self.d for encounter in c.encounters if encounter < self.time]))

        spreading_activation = self.get_spreading_activation_from_goal(chunk)

        return baselevel_activation + spreading_activation + self.noise(self.s)


    def get_latency(self, chunk):
        """
        Get the retrieval latency of the specified chunk at the current time.
        """
        activation = self.get_activation(chunk)
        return self.lf * math.exp(-self.le * activation)


    def noise(self, s):
        """
        Generate activation noise by drawing a value from a logistic distribution with mean 0 and scale s.
        """
        return 0


    def get_spreading_activation_from_goal(self, chunk):
        """
        Calculate the amount of spreading activation from the goal buffer to the specified chunk.
        """

        if self.goal is None:
            return 0

        if type(self.goal) is Chunk:
            total_slots = len(self.goal.slots)
            matching_slots = 0
            for slot, value in self.goal.slots.items():
                if value in chunk.slots.values():
                    matching_slots += 1
        
        if total_slots == 0:
            return 0

        return matching_slots * (self.ga / total_slots)


    def __str__(self):
        return "\n=== Model ===\n" \
        "Time: " + str(self.time) + " s \n" \
        "Goal:" + str(self.goal) + "\n" \
        "DM:" + "\n".join([str(c) for c in self.dm]) + "\n"