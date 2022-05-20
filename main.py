from __future__ import annotations
import mido
import numpy as np
import sys
import os
import copy
import music21
from matplotlib import pyplot as plt

class Scale:
    """
    Representation of a musical scale.
    """
    scale_chords = dict()
    tonic_pairs = {0: ('C', 'Am'), 1: ('G', 'Em'), 2: ('D', 'A#m'), 3: ('A', 'F#m'), 4: ('E', 'C#m'), 5: ('B', 'G#m'),
                   6: ('F#', 'D#m')}

    def __init__(self, scale_chords: list[Chord]):
        self.chords = scale_chords

    @classmethod
    def generate_major_scales(cls):
        """
            Generates major scales according to the tone signature of major scales.
        """
        # T -- T -- ST -- T -- T -- T -- ST
        major_scale = [2, 2, 1, 2, 2, 2, 1]
        major_scale_minor_chords = [2, 3, 6]
        major_scale_dim_step = 7
        for note_char in Note.NOTES:
            root = Note(note_char)
            scale_list = []
            for i, shift in enumerate(major_scale):
                _ch = None
                if i+1 in major_scale_minor_chords:
                    _ch = Chord.get_minor_chord(root)
                elif i+1 == major_scale_dim_step:
                    _ch = Chord.get_dim(root)
                else:
                    _ch = Chord.get_major_chord(root)
                scale_list.append(_ch)
                root = root + shift
            cls.scale_chords[note_char] = scale_list

    @classmethod
    def generate_minor_scales(cls):
        """
            Generates minor scales according to the tone signature of minor scales.
        """
        # Tone – Semitone – Tone – Tone – Semitone – Tone – Tone
        minor_scale = [2, 1, 2, 2, 1, 2, 2]
        minor_scale_minor_chords = [1, 4, 5]
        minor_scale_dim_step = 2
        for note_char in Note.NOTES:
            root = Note(note_char)
            scale_list = []
            for i, shift in enumerate(minor_scale):
                _ch = None
                if i+1 in minor_scale_minor_chords:
                    _ch = Chord.get_minor_chord(root)
                elif i+1 == minor_scale_dim_step:
                    _ch = Chord.get_dim(root)
                else:
                    _ch = Chord.get_major_chord(root)
                scale_list.append(_ch)
                root = root + shift

            cls.scale_chords[note_char + 'm'] = scale_list

    @classmethod
    def generate_scales(cls):
        """
            Generates all scales.
        """
        Scale.generate_minor_scales()
        Scale.generate_major_scales()

    @classmethod
    def good_chords(cls, tonic: Chord) -> list[Chord]:
        """
        Returns a list of good chords matching the scale with a given tonic (or, equivalently, a given key)
        :param tonic: tonic chord
        :return: a list of matching (consonant) (good) chords.
        """
        if len(Scale.scale_chords) == 0:
            Scale.generate_scales()
        return Scale.scale_chords[tonic.name]

    @classmethod
    def good_chords_str(cls, tonic_name: str) -> list[Chord]:
        if len(Scale.scale_chords) == 0:
            Scale.generate_scales()
        return Scale.scale_chords[tonic_name]

    @classmethod
    def from_key(cls, key: Chord) -> Scale:
        """
        Creates a Scale object using key chord (according to the lookup table)
        :param key: key (tonic) chord
        :return: a new Scale object
        """
        if key.type == 'major':
            return cls(Scale.good_chords(key))
        elif key.type == 'minor':
            return cls(Scale.good_chords(key))
        else:
            assert False

    def tonic(self) -> Chord:
        """
        Alias for the tonic chord of a scale.
        :return: Tonic chord.
        """
        return self.chords[0]

    def mediant(self) -> Chord:
        """
        Alias for the mediant chord of a scale.
        :return: Mediant chord.
        """
        return self.chords[2]

    def subdominant(self) -> Chord:
        """
        Alias for the subdominant chord of a scale.
        :return: Subdominant chord.
        """
        return self.chords[3]

    def dominant(self) -> Chord:
        """
        Alias for the dominant chord of a scale.
        :return: Dominant chord.
        """
        return self.chords[4]

    def is_sdt(self, seq: list[Chord]) -> bool:
        """
        Checks if the sequence of chords is a Subdominant-Dominant-Tonic progression
        :param seq: sequence of chords
        :return: True if the sequence is S-D-T, False otherwise
        """
        if len(seq) == 3:
            if seq[0] == self.subdominant() and seq[1] == self.dominant() and seq[2] == self.tonic():
                return True
        return False

    def is_st(self, seq: list[Chord]):
        """
        Checks if the sequence of chords is a Subdominant-Tonic progression
        :param seq: sequence of chords
        :return: True if the sequence is S-T, False otherwise
        """
        if len(seq) == 2:
            if seq[0] == self.subdominant() and seq[1] == self.tonic():
                return True
        return False

    def is_dt(self, seq: list[Chord]):
        """
        Checks if the sequence of chords is a Dominant-Tonic progression
        :param seq: sequence of chords
        :return: True if the sequence is D-T, False otherwise
        """
        if len(seq) == 2:
            if seq[0] == self.dominant() and seq[1] == self.tonic():
                return True
        return False


class Accompaniment:
    """
    Accompaniment representation
    """
    def __init__(self, chords: list[Chord]):
        self.chords = chords
        self.fitness = 0

    def __repr__(self):
        return str(self.fitness)

    def to_midi(self, melody_msgs) -> mido.MidiTrack:
        """
        Converts accompaniment chords to MIDI track.
        :param: melody_msgs: MIDI messages of main melody (for proper timings and synchronization)
        :return: a MIDI track with chords of the accompaniment
        """
        chord_track = mido.MidiTrack()
        c_ind, m_ind = 0, 0
        while m_ind < len(melody_msgs):
            note_msg = melody_msgs[m_ind]
            m_ind += 1
            if note_msg.type not in ['note_on', 'note_off']:
                if note_msg.type == 'track_name':
                    chord_track.append(mido.MetaMessage(type='track_name', name='Accompaniment (AI)', time=0))
                else:
                    chord_track.append(note_msg)
                continue
            chord = self.chords[c_ind]
            note_on = True if note_msg.type == 'note_on' else False
            _time = note_msg.time
            chord_messages = chord.to_midi_messages(note_on=note_on, time=note_msg.time, velocity=melody.notes[0].velocity-10)
            for chm in chord_messages:
                chm.time = _time
                _time = 0
                chord_track.append(chm)
            if not note_on:
                c_ind += 1
        #chord_track.append(mido.MetaMessage('end_of_track', time=0))
        return chord_track


class Melody:
    """
    Melody representation.
    """
    def __init__(self, midi_filename):
        """
        Constructor method.
        :param: midi_filename: name of input .mid file
        """
        # parse midi file
        self.filename = midi_filename
        self.midi_file, self.midi_messages, self.notes = load_midi_file(midi_filename)
        # find minimal octave used and tonality of the melody
        self.min_octave, self.key_pair, key = self.analyze_melody()
        self.scale = Scale.from_key(key)
        # generate all chords suitable for this melody
        self.chords_list = self.generate_chords(self.scale)
        self.chords_list = np.array(self.chords_list)
        # number of notes
        self.notes_num = len(self.notes)

    def analyze_melody(self) -> tuple:
        """
        Finds key and octave of the melody.
        :return:
        """
        assert len(self.notes) > 0
        sorted_by_octave = sorted(self.notes, key=lambda n: n.octave)
        min_octave = sorted_by_octave[0].octave
        sharp_notes = 0
        known_sharp = dict()
        notes_set = dict()
        for note in self.notes:
            notes_set[note.note_char] = note
            if '#' in note.note_char and note.note_char not in known_sharp:
                known_sharp[note.note_char] = note.note_char
                sharp_notes += 1
        key_pair = Scale.tonic_pairs[sharp_notes]
        ordered_notes_set = sorted(list(notes_set.values()), key=lambda x: x.midi_code)
        key = self.try_detect_key(ordered_notes_set, key_pair)
        return min_octave, key_pair, key

    def try_detect_key(self, notes: list[Note], candidates_pair: tuple) -> Chord:
        """
        Finds candidate for a key of the melody scale
        :param notes: a list of notes
        :param candidates_pair: two possible candidates (parallel keys)
        :return: tonic chord for a given scale
        """
        key = str(music21.converter.parse(self.filename).analyze('key')).upper()
        key_char = key[0]
        if key[1] == '#':
            key_char += '#'
        typ = key.split(' ')[0]
        if typ == 'major':
            return Chord.get_major_chord(Note(key_char))
        return Chord.get_minor_chord(Note(key_char))

    def generate_chords(self, scale: Scale) -> list:
        """
        Generates chords for a known melody scale up to minimal octave of the melody,
        including inverted, diminished and suspended chords.
        :param scale: A scale (corresponding to a key of a melody)
        :return: pool of chords in that scale
        """
        _chords = []
        for chord in self.scale.chords:
            # vary the octave from 0 to self.min_scale - 1
            # and generate some additional chords (inv, sus2/sus4)
            for i in range(self.min_octave):
                s_chord = chord.clone()
                s_chord.shift_chord(i)
                if s_chord.max_octave() >= self.min_octave:
                    break
                _chords.append(s_chord)
                inv1 = Chord.get_inv1(s_chord)
                inv2 = Chord.get_inv2(s_chord)
                if inv2.max_octave() < self.min_octave:
                    _chords.append(inv1)
                    _chords.append(inv2)
        return _chords


class Note:
    """
    Note representation.
    """
    NOTES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

    def __init__(self, note_char: str, octave=0, time_delta=0, midi_code=None, velocity=50):
        """
        Constructor method
        :param note_char: name of the note (for example, C or F#)
        :param octave: octave the note belongs to
        :param time_delta: (MIDI part) time passed since the previous action
        :param midi_code: midi code of the note
        :param velocity: how fast the note was played
        """
        assert(note_char in Note.NOTES)
        self.note_char = note_char
        self.time_delta = time_delta
        self.velocity = velocity
        if octave is not None:
            self.octave = octave
            self.midi_code = midi_code if midi_code is not None else Note.note_char_to_number(note_char, octave)

    @staticmethod
    def number_to_note_char(number: int) -> tuple:
        """
        Converts MIDI note number to a (note_char, octave) tuple.
        Based on this snippet: https://gist.github.com/devxpy/063968e0a2ef9b6db0bd6af8079dad2a
        :param number: MIDI number
        :return: a (note_char, octave) tuple.
        """
        assert 0 <= number <= 127
        octave = number // len(Note.NOTES)
        note = Note.NOTES[number % len(Note.NOTES)]
        return note, octave - 2  # octave correction when converting from midi

    @staticmethod
    def note_char_to_number(note: str, octave: int) -> int:
        """
        Converts (note_char, octave) to a MIDI note number.
        Based on this snippet: https://gist.github.com/devxpy/063968e0a2ef9b6db0bd6af8079dad2a
        :param note: note character
        :param octave: octave number
        :return: MIDI number
        """
        octave += 2  # midi octave correction when converting to midi
        assert note in Note.NOTES
        assert -2 <= octave <= 6
        note = Note.NOTES.index(note)
        note += (len(Note.NOTES) * octave)
        assert 0 <= note <= 127
        return note

    @classmethod
    def from_midi(cls, midi_msg: mido.Message, velocity=50):
        """
        Converts a MIDI message into a Note object.
        :param midi_msg:
        :param velocity:
        :return:
        """
        note, octave = Note.number_to_note_char(midi_msg.note)
        time_delta = midi_msg.time
        midi_code = midi_msg.note
        return cls(note, octave, time_delta, midi_code, velocity=velocity)

    def shift_octave(self, shamt: int) -> Note:
        """
        Creates a note that is given one shifted by shamt octaves
        :param shamt: shift amount
        :return: a Note object shifted as described.
        """
        return Note(self.note_char, octave=self.octave+shamt, time_delta=self.time_delta, midi_code=self.midi_code+12*shamt, velocity=self.velocity)

    # overloading of some operators
    def __str__(self):
        return f'{self.note_char}'

    def __repr__(self):
        return f'{self.note_char}{self.octave}'

    def __eq__(self, other: Note):
        if self is other:
            return True
        if self.note_char == other.note_char:
            return True
        return False

    def __add__(self, other: int):
        code = self.midi_code + other
        note, octave = Note.number_to_note_char(code)
        return Note(note, octave, self.time_delta, code)

    def __sub__(self, other: int):
        return self + -other

    def distance(self, other: Note) -> int:
        """
        Finds distance between two nodes in terms of their MIDI codes.
        :param other: note to be compared with
        :return: non-negative distance between midi codes of two notes.
        """
        return abs(self.midi_code - other.midi_code)

    def to_midi_msg(self, note_on, time=None, velocity=None) -> mido.Message:
        """
        Converts Note representation into a MIDI message
        :param note_on: should the note be turned on or turned off.
        :param time: time delta in ticks since the last message
        :param velocity: how fast the note was played
        :return: MIDI message with corresponding note.
        """
        msg_type = 'note_on'
        if not note_on:
            msg_type = 'note_off'
        if time is None:
            time = self.time_delta
        if velocity is None:
            velocity = self.velocity
        return mido.Message(msg_type, note=self.midi_code, velocity=velocity, time=time, channel=0)



class Chord:
    """
    Representation of a chord.
    """
    CHORD_TYPES = ['minor', 'major', 'inv1', 'inv2', 'dim', 'sus2', 'sus4', 'rest']

    def __init__(self, notes: list[Note], ctype=None, name=None):
        """
        Constructor method
        :param notes: A list of notes chord consists of
        :param ctype: types of a chord (see above)
        :param name: string representation of a chord, for example: F#m (without octave)
        """
        assert ctype in Chord.CHORD_TYPES
        self.type = ctype
        if ctype == 'rest':
            self.name = '<pause>'
            return
        assert len(notes) > 2
        self.notes = sorted(notes, key=lambda n: n.midi_code)
        self.name = name
        if self.name is None:
            self.name = self.get_name()

    def __eq__(self, other):
        """
        Equality operator
        Chords are equal if their names are the same (octave agnostic)
        """
        return str(self) == str(other)

    def __neq__(self, other):
        """
        Inequality operator
        Chords are inequal if their names differ (octave agnostic)
        """
        return str(self) != str(other)

    def get_name(self) -> str:
        """
        Constructs the standard name of a chord.
        :return: chord name
        """
        assert self.type in Chord.CHORD_TYPES
        name = str(self.root_note())
        if self.type == 'major':
            pass
        elif self.type == 'minor':
            name += 'm'
        else:
            name += self.type
        return name

    def shift_chord(self, shamt: int):
        """
            Shifts all notes of a chord by a particular value.
        :param shamt: shift amount
        """
        for i in range(len(self.notes)):
            self.notes[i] = self.notes[i].shift_octave(shamt)

    def root_note(self) -> Note:
        """
        :return: root note of a chord
        """
        return self.notes[0]

    def root(self) -> Note:
        """
        Shortcut for root_note
        :return: root note of a chord
        """
        return self.root_note()

    def highest_note(self) -> Note:
        """
        :return: highest note of a chord
        """
        return self.notes[-1]

    def clone(self) -> Chord:
        """
        :return: an exact clone (a deep copy) of the current object
        """
        _notes = copy.deepcopy(self.notes)
        return Chord(_notes, ctype=self.type)

    @classmethod
    def get_rest_chord(cls):
        """
        Creates a rest (pause) chord
        :return: a rest chord
        """
        return cls([], True)

    def __str__(self):
        """
        String conversion of a chord
        :return:
        """
        return f'{self.name}'

    def __repr__(self):
        """
        Representation of a chord
        :return:
        """
        return f'{self.name}{self.root_note().octave}'#{self.notes}'

    def to_midi_messages(self, note_on=True, time=None, velocity=None) -> list[mido.Message]:
        """
        Converts chord to MIDI messages
        :param note_on: whether the chord is turned on or turned off
        :param time: time delta since last message
        :param velocity: how fast the chord is played
        :return:
        """
        msgs = []
        for note in self.notes:
            msgs.append(note.to_midi_msg(note_on, time, velocity))
        return msgs

    def distance(self, other: Chord) -> int:
        """
        Gets distance between two chords (as a distance between their roots)
        :param other: another chord
        :return: non-negative distance between roots of these chords
        """
        return self.root().distance(other.root())

    def has_note(self, note: Note):
        """
        Checks the presence of a note ignoring the octave
        :param note: looked up note
        :return: True if the chord contains this note (octave-agnostic), False otherwise
        """
        for n in self.notes:
            if n.note_char == note.note_char:
                return True
        return False

    def max_octave(self) -> int:
        return max(self.notes, key=lambda x: x.octave).octave

    @staticmethod
    def get_minor_chord(root_note: Note) -> Chord:
        """
        Creates minor chord
        :param root_note: note to build chord from
        :return: new minor chord
        """
        return Chord([root_note, root_note + 3, root_note + 7], ctype='minor')

    @staticmethod
    def get_major_chord(root_note: Note) -> Chord:
        """
        Creates major chord
        :param root_note: note to build chord from
        :return: new major chord
        """
        return Chord([root_note, root_note + 4, root_note + 7], ctype='major')

    @staticmethod
    def get_inv1(chord: Chord) -> Chord:
        """
        Creates the first inverse of a chord
        :param chord: initial chord
        :return: first inverse of a chord
        """
        _notes = copy.deepcopy(chord.notes)
        _notes[0].octave += 1
        inv_chord = Chord(_notes, ctype='inv1')
        return inv_chord

    @staticmethod
    def get_inv2(chord: Chord) -> Chord:
        """
        Creates the second inverse of a chord
        :param chord: initial chord
        :return: second inverse of a chord
        """
        _notes = copy.deepcopy(chord.notes)
        _notes[0].octave += 1
        _notes[1].octave += 1
        inv_chord = Chord(_notes, ctype='inv2')
        return inv_chord

    @staticmethod
    def get_sus2(root_note: Note) -> Chord:
        """
        Creates 2nd suspended chord
        :param root_note: note to build chord from
        :return: suspended chord
        """
        return Chord([root_note, root_note + 2, root_note + 7], ctype='sus2') # too large difference in octaves between the melody and the chords

    @staticmethod
    def get_sus4(root_note: Note) -> Chord:
        """
        Creates 4th suspended chord
        :param root_note: note to build chord from
        :return: suspended chord
        """
        return Chord([root_note, root_note + 5, root_note + 7], ctype='sus4')

    @staticmethod
    def get_dim(root_note: Note) -> Chord:
        """
        Creates diminished chord
        :param root_note:
        :return: diminished chord
        """
        return Chord([root_note, root_note + 3, root_note + 6], ctype='dim')


def evolution(generations=2000, population_size=100, offsprings=80) -> tuple:
    """
    High-level evolutionary algorithm.
    Inspired by Lab 9 code and heavily reworked.
    :param generations: number of generations passed
    :param population_size: number of individuals in
    :param offsprings: number of children born by every generation
    :return: most fit individual
    """
    EVOLUTION_TRIALS = 2
    m_most_fit_individual = None
    m_most_fit_history = None
    for i in range(EVOLUTION_TRIALS):
        population = initial_population(population_size)
        most_fit_individual = None
        most_fit_history = []
        for i in range(generations):
            population = evolution_step(population, offsprings)
            most_fit_individual = population[-1]
            most_fit_history.append(most_fit_individual.fitness)
        if m_most_fit_individual is None or most_fit_individual.fitness > m_most_fit_individual.fitness:
            m_most_fit_individual = most_fit_individual
            m_most_fit_history = most_fit_history
    return m_most_fit_individual, generations, m_most_fit_history


def select_fittest(population: list[Accompaniment], new_individuals: list[Accompaniment]) -> list[Accompaniment]:
    """
    Perfroms selection of the fittest.
    :param population: (previous) population
    :param new_individuals: offspring
    :return: len(population) fittest individuals
    """
    population_size = len(population)
    population.extend(new_individuals)
    for p in population:
        p.fitness = individual_fitness(p)
    population.sort(key=lambda x: x.fitness)
    population = population[population_size:]
    return population


def evolution_step(population: list[Accompaniment], offsprings: int) -> list[Accompaniment]:
    """
    Evolution step for one generation (breeding and selection of the fittest)
    Inspired by Lab 9 code and heavily reworked.
    :param population: list of individuals
    :param offsprings: number of offsprings (or, equivalently, number of children)
    :return: updated population
    """
    mothers, fathers = split_population(population, offsprings)
    offsprings = []
    for mother, father in zip(mothers, fathers):
        child = mutate(crossover(mother, father))
        offsprings.append(child)
    new_generation = select_fittest(population, offsprings)
    return new_generation


def split_population(population: list[Accompaniment], n_offsprings: int) -> tuple:
    """
    Splits population into two parts (odd and even elements) for futher breeding.
    :param population: list of population.
    :param n_offsprings: number of offsprings to be produced
    :return: a tuple with population split into "mothers" and "fathers"
    """
    mothers = population[-2 * n_offsprings::2]
    fathers = population[-2 * n_offsprings+1::2]
    return mothers, fathers


def crossover(mother: Accompaniment, father: Accompaniment) -> Accompaniment:
    """
    Performs a one-point crossover.
    :param: mother: first breeding individual
    :param: father: second breeding individual
    :return: resulting individual
    """
    point = np.random.choice(list(range(len(mother.chords))), 1)[0]
    assert len(mother.chords) == len(father.chords)
    mother_head = mother.chords[:point].copy()
    father_tail = father.chords[point:].copy()

    mother_head.extend(father_tail)
    return Accompaniment(mother_head)


def mutate(child: Accompaniment) -> Accompaniment:
    """
    Performs mutation on a child.
    :param: child: individual to mutate
    :return: mutated individual
    """
    # swap mutation
    for i in range(1):
        i, j = np.random.choice(len(child.chords), 2, replace=False)
        child.chords[i], child.chords[j] = child.chords[j], child.chords[i]
        k = np.random.choice(len(child.chords), 1, replace=False)[0]
        child.chords[k] = np.random.choice(melody.chords_list, 1)[0]
    return child


def initial_population(population_size: int) -> list[Accompaniment]:
    """
    Generates initial population
    :param population_size: number of individuals in the population
    :return: population (list of individuals)
    """
    population = []
    for i in range(population_size):
        population.append(get_individual())
    return population


def get_individual() -> Accompaniment:
    """
    Creates random individual for the initial population.
    :return: invididual
    """
    individual = []
    for j in range(melody.notes_num):
        individual.append(np.random.choice(melody.chords_list))
    return Accompaniment(individual)


def individual_fitness(acc: Accompaniment) -> int:
    """
    Fitness function of the evolutionary algorithm
    :param acc: individual to evaluate
    :return: fitness score of the individual
    """
    BAD_CHORD_PENALTY = 100
    NOTE_NOT_IN_CHORD_PENALTY = 100000
    LARGE_DIST_PENALTY = 5
    LARGE_SPAN_PENALTY = 10
    LOW_CHORDS_PENALTY = 10
    INV_BONUS = 10
    TOO_MANY_INVS_PENALTY = 100
    PROG_BONUS = 10000
    WRONG_DIST_PENALTY = 100

    penalty = 0
    invs = 0
    sus = 0
    max_dist = 1
    min_octave = min(acc.chords, key=lambda c: c.root().octave).root().octave
    max_octave = max(acc.chords, key=lambda c: c.notes[2].octave).notes[2].octave
    max_octave_span = abs(max_octave - min_octave)
    for i in range(len(acc.chords)):
        chord = acc.chords[i]
        notes = [*chord.notes, melody.notes[i]]
        if notes[0].distance(notes[3]) > 24: # too large difference in octaves between the melody and the chords
            penalty -= 10*WRONG_DIST_PENALTY
        for i in range(3):
            if notes[i].distance(notes[i+1]) in [1, 2, 10, 11]:
                penalty -= WRONG_DIST_PENALTY
        # some inverted chords are good, but not too much
        if 'inv' in chord.type and i+1 < len(acc.chords) and 'inv' not in acc.chords[i+1].type:
            penalty += INV_BONUS
            invs += 1
        if 'sus' in chord.type:
            sus += 1
        # take a sequence of 3 or less subsequent chords
        seq = acc.chords[i:i+3:]
        s = melody.scale
        # reward good scale progressions
        if s.is_sdt(seq) or s.is_st(seq) or s.is_dt(seq):
            penalty += PROG_BONUS
        # chords that don't contain accompanying notes are very bad
        if not acc.chords[i].has_note(melody.notes[i]):
            penalty -= NOTE_NOT_IN_CHORD_PENALTY
        if i != len(acc.chords) - 1:
            dist = acc.chords[i].distance(acc.chords[i+1])
            if dist > max_dist:
                max_dist = dist
    # fraction of inverted chords should not exceed the rate, otherwise it will just be the melody one octave higher
    rate = 0.3
    if invs > int(rate*len(acc.chords)):
        penalty -= TOO_MANY_INVS_PENALTY*(invs - int(rate*len(acc.chords)))

    penalty -= max_dist*LARGE_DIST_PENALTY  # large distance between chords are undesirable
    if min_octave <= 0: # low notes are also undesirable
        penalty -= LOW_CHORDS_PENALTY + min_octave*LOW_CHORDS_PENALTY
    if sus > int(rate*len(acc.chords)): # large number of sus'es are undesirable
        penalty -= TOO_MANY_INVS_PENALTY*(invs - int(rate*len(acc.chords)))
    penalty -= max_octave_span*LARGE_SPAN_PENALTY
    return penalty


def get_notes(midi_file: mido.MidiFile) -> list[Note]:
    """
    Parses the MIDI messages.
    :param midi_file: MidiFile object containing tracks
    :return: a list of Note objects, converted from corresponding MidiTrack messages.
    """
    notes = []
    velocity = 50
    for midi_track in midi_file.tracks:
        for msg in midi_track:
            if msg.type == 'note_on':
                velocity = msg.velocity
            if msg.type == 'note_off':
                notes.append(Note.from_midi(msg, velocity))
    return notes


def load_midi_file(filename: str) -> tuple:
    """
    Parse input MIDI file.
    :param: filename: input file
    :return: a tuple of MidiFile object, Track object, Notes list
    """
    mid = mido.MidiFile(filename)
    # get array of notes
    track = mid.tracks[1]
    notes = get_notes(mid)
    return mid, track, notes


def save_accompaniment(acc: Accompaniment):
    """
    Saves accompaniment to a new MIDI file.
    :param acc:
    :return:
    """
    output_file = melody.midi_file
    output_file.type = 1  # synchronous tracks
    melody_track = output_file.tracks[1]
    chord_track = mido.MidiTrack()
    chord_track.extend(acc.to_midi(melody_track))
    output_file.tracks.append(chord_track)
    print('Resulting accompaniment: ')
    print(acc.chords)
    fname = os.path.basename(melody.filename)
    save_file = f'Output{fname}'
    print('Saved result as', save_file)
    output_file.save(save_file)


def draw_plot(generations: int, most_fit_history: list[Accompaniment]):
    """
    Draws plot based on fitness data
    :param generations: number of generations
    :param most_fit_history: most fit individual for each generation
    """
    plt.plot(range(generations), most_fit_history)
    plt.xlabel('generation')
    plt.ylabel('fitness of the most fit')
    plt.title(f'Fitness score of the most fittest individual through generations\n{os.path.basename(melody.filename)}')
    plt.grid(True)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <midi input file>')
        sys.exit(1)
    global melody
    melody = Melody(os.path.abspath(sys.argv[1]))
    accompaniment, generations, most_fit_history = evolution()
    save_accompaniment(accompaniment)
    print('Done')
    print(f"Generations passed: {generations}\nScore of the most fit: {accompaniment.fitness}")
    draw_plot(generations, most_fit_history)


if __name__ == '__main__':
    main()
