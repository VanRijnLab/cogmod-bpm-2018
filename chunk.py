class Chunk(object):

    def __init__(self, name, slots):
        self.name = name
        self.slots = slots
        self.encounters = []

    
    def add_encounter(self, time):
        """
        Add an encounter of this chunk at the specified time.
        """
        if time not in self.encounters:
            self.encounters.append(time)


    def __str__(self):
        slots = str(self.slots)
        encounters = str(self.encounters)
        return "Chunk " + self.name + "\n" \
        "Slots: " + slots + "\n" \
        "Encounters: " + encounters + "\n"
    