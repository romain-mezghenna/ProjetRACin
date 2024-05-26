from abc import ABC, abstractmethod

# Abstract class for agents
class Agent(ABC): 
    def __init__(self,model):
        self.model = model
        pass
    
    @abstractmethod
    def analyze(self):
        pass
