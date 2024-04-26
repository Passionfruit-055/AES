from IoV import *
from common.arguments import *
from agents.agents import Agents

POSSIBLE_KEY_LENGTHS = [16, 24, 32]

if __name__ == '__main__':
    args = get_common_args()

    env = IoV(args.n_vehicles, args.n_base_stations, args)

    agent = Agents(args)

    for episode in range(args.n_episodes):
        for t in range(args.n_steps):
            if not args.random_msg:
                raise NotImplementedError('Msg generating not implemented yet')

            key_lengths = agent.choose_random_actions(avail_actions=POSSIBLE_KEY_LENGTHS)

            env.set_key_length(key_lengths)
            env.step()
