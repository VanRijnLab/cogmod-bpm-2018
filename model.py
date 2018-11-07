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
    
    rt = -1.0 # retrieval threshols (:rt)
    
    mas = 2.0 # maxmimum spreading (:mas; default: 2.0)

    def __init__(self):
        self.time = 0
        self.goal = None
        self.dm = []

    def get_chunk(self, name):
        """
        Find the Chunk given its name
        """
        chunk_idx = [i for i, j in enumerate(self.dm) if j.name == name]
        if len(chunk_idx) == 0:
            return None
        else:
            return self.dm[chunk_idx[0]]

        
    def add_encounter(self, chunk):
        """
        Add an encounter of a specified chunk at the current time.
        If the chunk does not exist yet, create it first.
        """

        # If a chunk by this name does not yet exist, add it to DM
        if chunk.name not in [chunk.name for chunk in self.dm]:
            self.dm.append(chunk)
            # Calculate the fan by checking all slot values
            for ch1 in self.dm:
                if ch1.name in chunk.slots.values():
                    ch1.fan +=1
        
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
        rand = random.uniform(0.001,0.999)
        return s * math.log((1 - rand)/rand)


    def get_spreading_activation_from_goal(self, chunk):
        """
        Calculate the amount of spreading activation from the goal buffer to the specified chunk.
        """

        if self.goal is None:
            return 0

        if type(self.goal) is Chunk:
            spreading = 0.0
            for slot, value in self.goal.slots.items():
                ch1 = self.get_chunk(value)
                if ch1 != None and value in chunk.slots.values() and ch1.fan > 0:
                    spreading += max(0, self.mas - math.log(ch1.fan))
        return spreading * self.ga
    
    def match(self, chunk1, pattern):
        """
        Does chunk1 match pattern in chunk pattern?
        """
        for slot, value in pattern.slots.items():
            if not(slot in chunk1.slots and chunk1.slots[slot] == value):
                return False
        return True

    def retrieve(self, chunk):
        """
        Retrieve the chunk with the highest activation that matches the request in chunk
        Returns the chunk (or None) and the retrieval latency
        """
        retrieve_error = False
        bestMatch = None
        bestActivation = self.rt
        for ch in self.dm:
            act = self.get_activation(ch)
            if self.match(ch, chunk) and act > bestActivation:
                bestMatch = ch
                bestActivation = act
        if bestMatch == None:
            latency = self.lf * math.exp(-self.le * self.rt)
        else:
            latency = self.lf * math.exp(-self.le * bestActivation) # calculate it here to avoid a new noise draw
        return bestMatch, latency

    def __str__(self):
        return "\n=== Model ===\n" \
        "Time: " + str(self.time) + " s \n" \
        "Goal:" + str(self.goal) + "\n" \
        "DM:" + "\n".join([str(c) for c in self.dm]) + "\n"