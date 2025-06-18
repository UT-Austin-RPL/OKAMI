from robosuite.models.objects.primitive import CylinderObject

class SimpleBottle(CylinderObject):
    def __init__(self, name, size=(0.03, 0.1), rgba=(0.5, 0.5, 0.5, 1.0)):
        super().__init__(name=name, size=size, rgba=rgba)