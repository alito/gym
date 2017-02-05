"""
MAME interface, based on atari_env
"""

import os
import subprocess
from collections import defaultdict
import itertools

import numpy
from PIL import Image

import gym
from gym import error
from gym.spaces import Box, MultiDiscrete
from gym import utils
from gym.utils import seeding
from .connection import Socket

import logging
logger = logging.getLogger(__name__)


class MAMEEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    ResetFrames = 20 # The reset time is handled on the MAME side, so just skip a little bit here
    CoinToStartFrames = 60 
    StartToLiveFrames = 20 
    PressFrames = 4

    SwitchesOrder = ['left', 'right', 'up', 'down', 'button1', 'button2', 'button3',
    'button4', 'button5', 'button6', 'coin', 'player1']
    HorizontalDirectionRange = range(2)
    VerticalDirectionRange = range(2,4)
    ButtonsRange = range(4,10)
    MiscellaneousRange = range(10,12)

    class CommunicationError(Exception):
        """
        Generic error that happened somewhere in our communications
        """

    def __init__(self, game='galaxian', frameskip=(2, 5)):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, game=game, frameskip=frameskip)

        self.frameskip = frameskip
        self.viewer = None
        self.score = None

        self._seed()

        self.mamele_connection = Socket()
        socket_path = self.mamele_connection.start_server()
        self.mame = self._start_mame(game, socket_path)

        # we'll initialise these once we know what we are dealing with
        self.latest_image_as_string = None
        self.previous_score = self.score = 0
        self.images_size_in_bytes = 0
        self.action_space = None
        self.action_space_mappings = None
        self._separate_action_spaces = None
        self.observation_space = None
        self.width = None
        self.height = None
        self.game_over = True
        self.resetting = False

        self.number_of_switches = len(self.SwitchesOrder)
        self.nothing_pressed = '0' * self.number_of_switches # template for the switches to send, all unpressed
        self.action_to_description = {}

        self.last_received = False

        # wait for mame to connect
        self.mamele_connection.wait_for_connection()

        # we expect the mame module to send the size and the minimal button set
        self.receive_message()
        self.receive_message()


    def send_message(self, message):
        if not self.last_received:
            # the passthrough only receives once after sendin an update, so avoid the deadlock and 
            # receive the message first
            self.receive_message()

        self.mamele_connection.send(message)
        self.last_received = False

    def receive_message(self):
        import time
        try:
            # we have a fixed command size of the first four characters
            # should do for now

            command = self.mamele_connection.receive_bytes(4).lower()
            if command == 'size':
                # we have the size of the space. Initialise buffer and observation space
                size_description = self.mamele_connection.receive_until_character('\n')
                self._initialise_screen(size_description.strip())
            elif command == 'used':
                # get the switches that are used
                switches_used_description = self.mamele_connection.receive_until_character('\n')
                self._initialise_action_space(switches_used_description.strip())
            elif command == 'quit':
                logger.info("Got a quit from the environment")
                self.expected_quit()
            elif command == 'updt':
                # combo update of score and image
                score_description = self.mamele_connection.receive_until_character('\n')
                game_over_description = self.mamele_connection.receive_until_character('\n')
                self.latest_image_as_string = self.mamele_connection.receive_bytes(self.images_size_in_bytes)
                if not self.resetting:
                    # ignore score and game over status while we are resetting
                    self._set_score(score_description.strip())
                    self._set_game_over(game_over_description.strip())
                self.last_received = True


        except self.CommunicationError as error:
            logger.error("Something went wrong talking to mamele: %s" % error)
            self.unexpected_quit()


    def _initialise_screen(self, description):
        # we get sent something like 400x300 (widthxheight)

        parts = description.split('x')
        if len(parts) != 2:
            raise self.CommunicationError("Didn't get a size in width x height format")

        try:
            self.width = int(parts[0])
            self.height = int(parts[1])
            logger.info("Screen size: %sx%s" % (self.width, self.height))
        except ValueError as error:
            raise self.CommunicationError("Either width or height weren't integers")

        self._buffer = numpy.empty((self.height, self.width, 4), dtype=numpy.uint8)
        self.images_size_in_bytes = self.height * self.width * 4 # comes as BGRA
        self.observation_space = Box(low=0, high=255, shape=(self.height, self.width, 3))



    def _initialise_action_space(self, switches_used_description):

        # We get a 0 or a 1 for each of the switches sent back. eg '111111000011'
        # Treat each of the directions separately, and each separate from the buttons

        if len(switches_used_description) != len(self.SwitchesOrder):
            raise IOError("Got a description of the switches used of an unexpected length. Expected %s, got %s (%s)" % (len(self.SwitchesOrder), switches_used_description, len(switches_used_description)))

        spaces = defaultdict(list)

        for index, used in enumerate(switches_used_description):
            if used == '1':
                if index in self.HorizontalDirectionRange:
                    spaces['horizontal'].append(index)
                elif index in self.VerticalDirectionRange:
                    spaces['vertical'].append(index)
                elif index in self.ButtonsRange:
                    spaces[index].append(index)
                # the other two miscellaneous ones are coin insertion and start of player 1
                # we'll handle those 

        # define them in a specific order for .... reasons

        # each space is the size of the switches for it plus one for no press

        action_spaces = []
        self.action_space_mappings = []
        for name, switches in spaces.items():
            action_spaces.append([0, len(switches)])
            self.action_space_mappings.append(switches)

        self.action_space = MultiDiscrete(action_spaces)
        self._separate_action_spaces = action_spaces

        self.generate_switch_mapping()

    def _set_score(self, description):
        self.previous_score = self.score
        self.score = int(description)

    def _set_game_over(self, description):
        # we only set game over to True here
        if description == '1':
            logger.debug("Gameover")
            self.game_over = True

    def expected_quit(self):
        # mame-side expected quit
        self.mamele_connection.destroy()

    def unexpected_quit(self):
        # mame-side hang up unexpectedly

        self.mamele_connection.destroy()
        # now bail
        raise IOError("Could not connect to our module in mamele land")

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    def _start_mame(self, game, socket_path):

        this_directory = os.path.realpath(os.path.dirname(__file__))
        passthrough_module = os.path.join(this_directory, 'passthrough')

        # volume -32 and low samplerate 
        # for now since turning off sound properly crashes mame on linux at the moment (2017/01)
        command = ['mame64'] + [game] + '-nothrottle -window -noautosave -frameskip 0 -volume -32 -skip_gameinfo -noshow_le -noautoframeskip -use_le -learning_environment pythonbinding.so -le_options'.split()

        # -noshow_le 
        # le_options is one parameter, the python bindings of mamele split it into the module name,
        # and the rest. That rest is passed to the module which can do with it as it pleases
        command.append("%s %s" % (passthrough_module, socket_path))
        process = subprocess.Popen(command, stderr=subprocess.STDOUT, close_fds=True)

        return process

    def map_action_to_switch_presses(self, action):
        """
        Map the integer internal actions to keypresses
        """
        return self.action_to_description[tuple(action)]


    def _step(self, action):

        switch_presses = self.map_action_to_switch_presses(action)

        if isinstance(self.frameskip, int):
            steps = self.frameskip
        else:
            steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])

        for _ in range(steps):
            self.send_message("inpt %s\n" % switch_presses)


        reward = self.score - self.previous_score
        return self._get_obs(), reward, self.game_over, {}


    def _get_image(self):
        return numpy.asarray(Image.frombuffer("RGBA",(self.width, self.height), self.latest_image_as_string,'raw', ("BGRA",0,1)))

    @property
    def _n_actions(self):
        total = 1
        for subspace in self._separate_action_spaces:
            total *= len(subspace)
        return total

    def _get_obs(self):    
        return self._get_image()

    # return: (states, observations)
    def _reset(self):
        # reset the machine, insert a coin, press player 1

        self.resetting = True
        if not self.last_received:
            # make sure it's waiting for us
            self.receive_message()
        if not self.game_over:
            self.send_message('rest')
            self.skip(self.ResetFrames)
        self.insert_coin()
        self.skip(self.PressFrames)
        self.press_nothing()
        self.skip(self.CoinToStartFrames)        
        self.start_player1()
        self.skip(self.PressFrames)
        self.press_nothing()
        self.skip(self.StartToLiveFrames)
        self.game_over = False
        self.score = self.previous_score = 0
        self.resetting = False
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def _close(self):
        """
        Shutdown
        """
        self.send_message('quit\n')
        self.expected_quit()


    def insert_coin(self):
        self.send_message("inpt %s\n" % self.action_to_description['coin'])

    def start_player1(self):
        self.send_message("inpt %s\n" % self.action_to_description['player1'])

    def press_nothing(self):
        self.send_message("inpt %s\n" % self.nothing_pressed)

    def skip(self, frames):
        self.send_message('skip %d\n' % frames)        

    def generate_switch_mapping(self):
        """
        Pre-generate all switch configurations mapping to what we send back
        """
        SwitchIndex = {}
        for index, switch in enumerate(self.SwitchesOrder):
            SwitchIndex[switch] = index


        coin_button = SwitchIndex['coin']
        player1_button = SwitchIndex['player1']
        self.action_to_description['coin'] = ''.join('0' if index != coin_button else '1' for index in range(self.number_of_switches))
        self.action_to_description['player1'] = ''.join('0' if index != player1_button else '1' for index in range(self.number_of_switches))


        # iterate over all our action spaces and generate a description for each one
        all_spaces = [range(space[0], space[1]+1) for space in self._separate_action_spaces]
        for action in itertools.product(*all_spaces):
            switches = set()
            # action is a tuple with one action from each space
            for component, mapping in zip(action, self.action_space_mappings):
                if component == 0:
                    # 0 is the no-press option
                    continue
                else:
                    switches.add(mapping[component-1])
            # create the description
            self.action_to_description[action] = ''.join('1' if index in switches else '0' for index in range(self.number_of_switches))

    # def save_state(self):

    # def load_state(self):

    # def clone_state(self):

    # def restore_state(self, state):

