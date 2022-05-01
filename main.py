from __future__ import annotations
import mido
import numpy as np
import sys
import copy
import music21
from matplotlib import pyplot as plt

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)



tonic_pairs = {0: ('C', 'Am'), 1: ('G', 'Em'), 2: ('D', 'A#m'), 3: ('A', 'F#m'), 4: ('E', 'C#m'), 5: ('B', 'G#m'),
               6: ('F#', 'D#m')}


class Scale:
    scale_chords_major = {
        'C': ('C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim'),
        'D': ('D', 'A#m', 'A', 'F#m', 'G', 'Em'),
        'E': ('E', 'F#m', 'G#m', 'A', 'B', 'C#m', 'D#dim'),
        'F': ('F', 'Gm', 'Am', 'A#', 'C', 'Dm'),
        'G': ('G', 'Em', 'C', 'Am', 'D'),
        'A': ('A', 'F#m', 'C#m', 'E', 'D', 'A#m'),
        'B': ('B', 'G#m', 'F#', 'D#m', 'E', 'C#m')
    }

    scale_chords_minor = {
        'C#m': ()
    }

    def __init__(self, scale_chords):
        self.scale_chords = scale_chords

    @classmethod
    def generate_major_scales(cls):
        # T -- T -- ST -- T -- T -- T -- ST
        major_scale = [2, 2, 1, 2, 2, 2, 1]
        major_scale_minor_chords = [2, 3, 6]
        major_scale_dim_step = 7
        for note_char in NOTES:
            root = Note(note_char)
            scale_list = []
            for i, shift in enumerate(major_scale):
                _ch = None
                if i+1 in major_scale_minor_chords:
                    _ch = Melody.get_minor_chord(root)
                elif i+1 == major_scale_dim_step:
                    _ch = Melody.get_dim(root)
                else:
                    _ch = Melody.get_major_chord(root)
                scale_list.append(_ch)
                root = root + shift
            scale = Scale(scale_list)
            cls.scale_chords_major[note_char] = Scale

    @classmethod
    def generate_minor_scales(cls):
        # Tone – Semitone – Tone – Tone – Semitone – Tone – Tone
        minor_scale = [2, 1, 2, 2, 1, 2, 2]
        minor_scale_minor_chords = [1, 4, 5]
        minor_scale_dim_step = 2
        for note_char in NOTES:
            root = Note(note_char)
            scale_list = []
            for i, shift in enumerate(minor_scale):
                _ch = None
                if i+1 in minor_scale_minor_chords:
                    _ch = Melody.get_minor_chord(root)
                elif i+1 == minor_scale_dim_step:
                    _ch = Melody.get_dim(root)
                else:
                    _ch = Melody.get_major_chord(root)
                scale_list.append(_ch)
                root = root + shift

            cls.scale_chords_minor[note_char] = scale_list


    @classmethod
    def from_key(cls, key: Chord) -> Scale:
        """
        Creates a Scale object using key chord (according to the lookup table)
        :param key: key (tonic) chord
        :return: a new Scale object
        """
        if key.type == 'major':
            return cls(Scale.scale_chords_major[key.name])
        elif key.type == 'minor':
            return cls(Scale.scale_chords_minor[key.name])
        else:
            assert False

    def tonic(self):
        return self.scale_chords[0]

    def mediant(self):
        return self.scale_chords[2]

    def subdominant(self):
        return self.scale_chords[3]

    def dominant(self):
        return self.scale_chords[4]



def good_chords(tonic: str) -> str:
    assert 'm' not in tonic
    return Scale.scale_chords_major[tonic]


class Accompaniment:
    def __init__(self, chords: list[Chord]):
        self.chords = chords
        self.fitness = 0

    def __repr__(self):
        return str(self.fitness)

    def to_midi(self, melody_msgs) -> mido.MidiTrack:
        chord_track = mido.MidiTrack()
        #chord_track.append(mido.Message('program_change', program=0, time=0))
        c_ind, m_ind = 0, 0
        while m_ind < len(melody_msgs):
            note_msg = melody_msgs[m_ind]
            m_ind += 1
            if note_msg.type not in ['note_on', 'note_off']:
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
    def __init__(self, midi_filename):
        # parse midi file
        self.filename = midi_filename
        self.midi_file, self.midi_messages, self.notes = load_midi_file(midi_filename)
        # find minimal octave used and tonality of the melody
        self.min_octave, self.key_pair, self_key = self.analyze_melody()
        # generate all chords suitable for this melody
        self.chords_dict, self.chords_list = self.generate_chords()
        self.chords_list = np.array(self.chords_list)
        # number of notes
        self.notes_num = len(self.notes)


    def analyze_melody(self):
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
        key_pair = tonic_pairs[sharp_notes]
        ordered_notes_set = sorted(list(notes_set.keys()), key=lambda x: x.midi_code)
        key = self.try_detect_key(ordered_notes_set, key_pair)
        return min_octave, key_pair, key

    def try_detect_key(self, notes: list[Note], candidates_pair: tuple) -> Chord:
        """
        Finds the most probable candidate for a key of melody scale
        :param notes: a list of notes
        :param candidates_pair: two possible candidates (parallel keys)
        :return: tonic chord for a given scale
        """
        if True:
            key = str(music21.converter.parse(melody.input_file).analyze('key').upper())
            key_char = key[0]
            if key[1] == '#':
                key_char += '#'
            typ = key.split(' ')[0]
            if typ == 'major':
                return Melody.get_major_chord(Note(key_char, None))
            return Melody.get_minor_chord(Note(key_char, None))
        else:
            scoreA = 0; scoreB = 0
            keyA = candidates_pair[0]
            scaleA = Scale.from_key(keyA)
            keyB = candidates_pair[1]
            scaleB = Scale.from_key(keyB)
            # criterion 1: last note of the melody end with stable chord of the scale
            # (tonic, mediant, dominant)
            stableA = [scaleA.tonic().root_note(), scaleA.mediant().root_note(), scaleA.dominant().root_node()]
            stableB = [scaleB.tonic().root_note(), scaleB.mediant().root_note(), scaleB.dominant().root_node()]
            if notes[-1] in stableA:
                scoreA += 100
            if notes[-1] in stableB:
                scoreB += 100
            freqA = 0
            freqB = 0
            for note in notes:
                if note == keyA.root_note():
                    freqA += 1
                elif note == keyB.root_note():
                    freqB += 1



    @staticmethod
    def get_minor_chord(base: Note) -> Chord:
        return Chord([base, base+3, base+7], type='minor')

    @staticmethod
    def get_major_chord(base: Note) -> Chord:
        return Chord([base, base+4, base+7], type='major')

    @staticmethod
    def get_inv1(chord: Chord) -> Chord:
        _notes = copy.deepcopy(chord.notes)
        _notes[0].octave += 1
        inv_chord = Chord(_notes, type='inv1')
        return inv_chord

    @staticmethod
    def get_inv2(chord: Chord) -> Chord:
        _notes = copy.deepcopy(chord.notes)
        _notes[0].octave += 1
        _notes[1].octave += 1
        inv_chord = Chord(_notes, type='inv2')
        return inv_chord

    @staticmethod
    def get_sus2(base: Note) -> Chord:
        return Chord([base, base+2, base+7], type='sus2')

    @staticmethod
    def get_sus4(base: Note) -> Chord:
        return Chord([base, base+5, base+7], type='sus4')

    @staticmethod
    def get_dim(base: Note) -> Chord:
        return Chord([base, base+3, base+6], type='dim')


    def generate_chords(self) -> tuple:
        _chords_dict = dict()
        _chords = []
        root_note = Note('C', 1) # consider all possible notes up to octave the melody itself uses
        while root_note.octave != self.min_octave:
            minor_chord = Melody.get_minor_chord(root_note)
            major_chord = Melody.get_major_chord(root_note)
            if major_chord.highest_note().octave == self.min_octave:
                break
            major_chords.append(major_chord)
            minor_chords.append(minor_chord)
            _chords_dict[major_chord.name] = major_chord
            _chords_dict[minor_chord.name] = minor_chord
            _chords.append(major_chord)
            _chords.append(minor_chord)

            root_note = root_note + 1
        return _chords_dict, _chords


class Note:
    def __init__(self, note_char: str, octave=0, time_delta=0, midi_code=None, velocity=50):
        assert(note_char in NOTES)
        self.note_char = note_char
        self.time_delta = time_delta
        self.velocity = velocity
        if octave is not None:
            self.octave = octave
            self.midi_code = midi_code if midi_code is not None else note_to_number(note_char, octave)

    @classmethod
    def from_midi(cls, midi_msg: mido.Message, velocity=50):
        assert(midi_msg.type == 'note_off')
        note, octave = number_to_note(midi_msg.note)
        time_delta = midi_msg.time
        midi_code = midi_msg.note
        return cls(note, octave, time_delta, midi_code, velocity=velocity)

    def __str__(self):
        #return f'{self.note}{self.octave}'
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
        note, octave = number_to_note(code)
        return Note(note, octave, self.time_delta, code)

    def __sub__(self, other: int):
        return self + -other

    def distance(self, other: Note) -> int:
        return abs(self.midi_code - other.midi_code)

    def to_midi_msg(self, note_on, time=None, velocity=None):
        msg_type = 'note_on'
        if not note_on:
            msg_type = 'note_off'
        if time is None:
            time = self.time_delta
        if velocity is None:
            velocity = self.velocity
        return mido.Message(msg_type, note=self.midi_code, velocity=velocity, time=time, channel=0)


def number_to_note(number: int) -> tuple:
    octave = number // NOTES_IN_OCTAVE
    note = NOTES[number % NOTES_IN_OCTAVE]
    return note, octave-2

def note_to_number(note: str, octave: int) -> int:
    octave += 2
    assert note in NOTES
    assert octave in OCTAVES
    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)
    assert 0 <= note <= 127
    return note


class Chord:
    CHORD_TYPES = ['minor', 'major', 'inv1', 'inv2', 'dim', 'sus2', 'sus4', 'rest']

    def __init__(self, notes: list[Note], type=None, name=None):
        assert type in Chord.CHORD_TYPES
        self.type = type
        if type == 'rest':
            self.name = '<pause>'
            return
        assert len(notes) > 2
        self.notes = sorted(notes, key=lambda n: n.midi_code)
        self.name = name
        if self.name is not None:
            self.name = self.get_name()

    def __eq__(self, other):
        return str(self) == str(other)

    def get_name(self) -> str:
        assert type in Chord.CHORD_TYPES
        name = str(self.root_note())
        if self.type == 'major':
            pass
        elif self.type == 'minor':
            name += 'm'
        else:
            name += self.type
        return name



    def root_note(self) -> Note:
        return self.notes[0]

    def highest_note(self) -> Note:
        return self.notes[-1]

    def is_minor(self):
        return self.notes[0].distance(self.notes[1]) == 3

    def clone(self) -> Chord:
        _notes = copy.deepcopy(self.notes)
        return Chord(_notes, type=self.type)

    @classmethod
    def detect_type(notes) -> str:
        # not really needed
        pass

    @classmethod
    def get_rest_chord(cls):
        return cls([], True)

    def __str__(self):
        return f'{self.name}{self.notes}'

    def __repr__(self):
        return f'{self.name}{self.notes}'

    def to_midi_messages(self, note_on=True, time=None, velocity=None) -> list[mido.Message]:
        msgs = []
        for note in self.notes:
            msgs.append(note.to_midi_msg(note_on, time, velocity))
        return msgs

    def tonic(self) -> Note:
        return self.notes[0]

    def distance(self, other: Chord) -> int:
        return self.tonic().distance(other.tonic())

    def has_note(self, note: Note):
        for n in self.notes:
            if n.note_char == note.note_char:
                return True
        return False


chords_dict = dict()
minor_chords = []
major_chords = []


def evolution(generations=2000, population_size=100, n_offsprings=80) -> Accompaniment:
    population = initial_population(population_size)
    most_fit_individual = None
    most_fit_history = []
    for i in range(generations):
        population = evolution_step(population, n_offsprings)
        most_fit_individual = population[-1]
        most_fit_history.append(most_fit_individual.fitness)
    plt.plot(range(generations), most_fit_history)
    plt.xlabel('generation')
    plt.ylabel('fitness of the most fit')
    plt.title('Fitness score of the most fittest individual through generations\n')
    plt.grid(True)
    plt.show()
    return most_fit_individual


def replace_population(population: list[Accompaniment], new_individuals: list[Accompaniment]):
    population_size = len(population)
    population.extend(new_individuals)
    for p in population:
        p.fitness = individual_fitness(p)
    population.sort(key=lambda x: x.fitness)
    population = population[population_size:]
    return population


def evolution_step(population: list[Accompaniment], n_offsprings: int):
    mothers, fathers = split_population(population, n_offsprings)
    offsprings = []
    for mother, father in zip(mothers, fathers):
        child = mutate(crossover(mother, father))
        offsprings.append(child)
    #prev_most_fit = population[-1]
    #offsprings.append(prev_most_fit)
    new_generation = replace_population(population, offsprings)
    return new_generation


def split_population(population: list[Accompaniment], n_offsprings: int) -> tuple:
    mothers = population[-2 * n_offsprings::2]
    fathers = population[-2 * n_offsprings+1::2]
    return mothers, fathers


def crossover(mother: Accompaniment, father: Accompaniment) -> Accompaniment:
    point = np.random.choice(list(range(len(mother.chords))), 1)[0]
    assert len(mother.chords) == len(father.chords)
    mother_head = mother.chords[:point].copy()
    mother_tail = mother.chords[point:].copy()
    father_tail = father.chords[point:].copy()

    """
    mapping = {father_tail[i]: mother_tail[i] for i in range(len(mother_tail))}

    for i in range(len(mother_head)):
        while mother_head[i] in father_tail:
            mother_head[i] = mapping[mother_head[i]]
    """
    mother_head.extend(father_tail)
    return Accompaniment(mother_head)


def mutate(child: Accompaniment) -> Accompaniment:
    # swap mutation
    for i in range(1):
        i, j = np.random.choice(len(child.chords), 2, replace=False)
        child.chords[i], child.chords[j] = child.chords[j], child.chords[i]
        k = np.random.choice(len(child.chords), 1, replace=False)[0]
        child.chords[k] = np.random.choice(melody.chords_list, 1)[0]
    return child


def initial_population(population_size: int) -> list[Accompaniment]:
    population = []
    for i in range(population_size):
        population.append(get_individual())
    return population


def get_individual() -> Accompaniment:
    individual = []
    for j in range(melody.notes_num):
        individual.append(np.random.choice(melody.chords_list))
    return Accompaniment(individual)


def individual_fitness(acc: Accompaniment):
    BAD_CHORD_PENALTY = 100
    NOTE_NOT_IN_CHORD_PENALTY = 100000
    LARGE_DIST_PENALTY = 5
    LARGE_SPAN_PENALTY = 10
    LOW_CHORDS_PENALTY = 10
    penalty = 0
    max_dist = 1
    min_octave = min(acc.chords, key=lambda c: c.notes[0].octave).notes[0].octave
    max_octave = max(acc.chords, key=lambda c: c.notes[2].octave).notes[2].octave
    max_octave_span = abs(max_octave - min_octave)
    for i in range(len(acc.chords)):
        if not acc.chords[i].has_note(melody.notes[i]): # idea 1: chords that don't contain accompanying notes are very bad
            penalty -= NOTE_NOT_IN_CHORD_PENALTY
        if acc.chords[i].name not in good_chords(melody.key_pair[0]):
            penalty -= BAD_CHORD_PENALTY
        if i != len(acc.chords) - 1:
            dist = acc.chords[i].distance(acc.chords[i+1])
            if dist > max_dist:
                max_dist = dist
    penalty -= max_dist*LARGE_DIST_PENALTY # idea 2: distance between chords are undesirable
    if min_octave <= 0:
        penalty -= LOW_CHORDS_PENALTY + min_octave*LOW_CHORDS_PENALTY
    penalty -= max_octave_span*LARGE_SPAN_PENALTY
    return penalty


def get_notes(midi_track) -> list:
    notes = []
    velocity = 50
    for msg in midi_track:
        if msg.type == 'note_on':
            velocity = msg.velocity
        if msg.type == 'note_off':
            notes.append(Note.from_midi(msg, velocity))
    return notes


def load_midi_file(filename: str):
    mid = mido.MidiFile(filename)
    # get array of notes
    track = mid.tracks[1]
    notes = get_notes(track)
    return mid, track, notes


def print_midi_file(mid):
    for i, track in enumerate(mid.tracks):
        print('Track {}:'.format(i))
        for msg in track:
            print(msg)


def test():
    note = Note('C', 3)
    #print(chord)
    #print(chord.to_midi_messages(note_on=True))


def save_accompaniment(acc: Accompaniment):
    output_file = melody.midi_file
    output_file.type = 1
    melody_track = output_file.tracks[1]
    #midi_data = melody.midi_file.tracks[0][0:2]
    #melody_track = mido.MidiTrack()
    chord_track = mido.MidiTrack()
    #melody_track.extend(midi_data)
    #melody_track.extend(melody.midi_file.tracks[1])
    chord_track.extend(acc.to_midi(melody_track))
    #output_file.tracks.append(midi_data)
    #output_file.tracks.append(melody_track)
    output_file.tracks.append(chord_track)
    output_file.save(f'output_{sys.argv[1]}.mid')


melody = None
def main():
    test()
    global melody
    melody = Melody(sys.argv[1])
    accompaniment = evolution()
    save_accompaniment(accompaniment)
    print('done')

if __name__ == '__main__':
    main()