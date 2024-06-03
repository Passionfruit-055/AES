class FixedAgent(object):
    def __init__(self, args, action):
        self.args = args
        self.action = action

    def choose_action(self, state):
        return self.action, state

    def learn(self):
        pass

    def store_transition(self, state, action, reward, state_):
        pass
