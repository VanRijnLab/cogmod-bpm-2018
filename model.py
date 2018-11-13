from chunk import Chunk
import math
import random

class Model(object):

    # Model parameters

    ga = 1.0 # spreading activation from the goal (:ga; default: 1.0)
    mas = 2.0 # maxmimum spreading (:mas; default: 2.0)

    d = 0.5 # decay (:bll; default: 0.5)
    s = 0 # scale of activation noise (:ans; default: 0)

    lf = 1.0 # latency factor (:lf; default: 1.0)
    le = 1.0 # latency exponent (:le; default: 1.0)
    
    rt = -1.0 # retrieval threshold (:rt; default: 0.0)

    t = 0.2 # blending noise parameter from Taatgen & van Rijn (2011)
    

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


    def update_fan(self):
        """
        Recalculate the fan of all chunks in DM.
        """
        for ch1 in self.dm:
            ch1.fan = 0
            for ch2 in self.dm:
                if ch1.name in ch2.slots.values():
                    ch1.fan += 1 # ch2 refers to ch1


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

        # Add slot values as singleton chunks
        for v in chunk.slots.values():
            s = Chunk(name = v, slots = {})
            self.add_encounter(s)

        # Recalculate the fan
        self.update_fan()




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
            total_slots = 0
            for value in self.goal.slots.values():
                total_slots += 1
                ch1 = self.get_chunk(value)
                if ch1 != None and value in chunk.slots.values() and ch1.fan > 0:
                    spreading += max(0, self.mas - math.log(ch1.fan))
        
        if total_slots == 0:
            return 0

        return spreading * (self.ga / total_slots)


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


    def get_retrieval_probability(self, chunk, pattern):
        """
        Returns the probability of retrieving a specific chunk that matches the specified pattern,
        given its activation and the activation of the other matching chunks
        """
        activations = dict([(ch, self.get_activation(ch))for ch in self.dm if self.match(ch, pattern)])
        return math.exp(activations[chunk] / self.t)  / sum([math.exp(a / self.t) for a in activations.values()])


    def retrieve_blended_trace(self, pattern, slot):
        """
        Returns a blend of the requested slot value from all chunks in DM that match the specified pattern, weighted by their activation
        """

        eligible_chunks = [ch for ch in self.dm if self.match(ch, pattern) and slot in ch.slots and ch.slots[slot]]
        
        if not eligible_chunks:
            return None

        return sum([self.get_retrieval_probability(ch, pattern) * ch.slots[slot] for ch in eligible_chunks])


    def __str__(self):
        return "\n=== Model ===\n" \
        "Time: " + str(self.time) + " s \n" \
        "Goal:" + str(self.goal) + "\n" \
        "DM:" + "\n".join([str(c) for c in self.dm]) + "\n"