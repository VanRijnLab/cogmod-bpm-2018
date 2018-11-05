import math

class DM(object):

    # Declarative memory parameters
    param_d = 0.5 # decay (:bll; default: 0.5)
    param_lf = 1.0 # latency factor (:lf; default: 1.0)
    param_le = 1.0 # latency exponent (:le; default: 1.0)


    def __init__(self):
        self.chunks = {}


    def __str__(self):
        return str(self.chunks)


    def add_encounter(self, content, time):
        # If a chunk with this content already exists, add an encounter, otherwise create a new chunk
        if content in self.chunks:
            if time not in self.chunks[content]:
                self.chunks[content].append(time)
        else:
            self.chunks[content] = [time]


    def get_baselevel_activation(self, chunk, current_time):
        if chunk not in self.chunks:
            raise KeyError("Chunk %s does not exist in DM" % (str(chunk)))

        encounters = self.chunks[chunk]

        if current_time <= min(encounters):
            raise ValueError("Chunk %s not encountered at or before time %s" % (str(chunk), str(current_time)))
        else:
            return math.log(sum([(current_time - encounter) ** -self.param_d for encounter in encounters if encounter < current_time]))


    def get_latency(self, chunk, current_time):
        if chunk not in self.chunks:
            raise KeyError("Chunk %s does not exist in DM" % (str(chunk)))

        activation = self.get_baselevel_activation(chunk, current_time)

        return self.param_lf * math.exp(-self.param_le * activation)