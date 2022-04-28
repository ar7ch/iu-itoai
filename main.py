from __future__ import annotations
import mido
import numpy as np

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)
good_chords_major = {'C': ('C', 'Dm', 'Em', 'F', 'G', 'Am', 'B'),
                     'C#': ()
                     }

tonic_pairs = {0: ('C', 'Am'), 1: ('G', 'Em'), 2: ('D', 'A#m'), 3: ('A', 'F#m'), 4: ('E', 'C#m'), 5: ('B', 'G#m'), 6: ('F#', 'D#m')}

class Accompaniment:
    def __init__(self, chords: list[Chord]):
        self.chords = chords

class Melody:
    def __init__(self, midi_filename):
        # parse midi file
        self.midi_file, self.midi_messages, self.notes = load_midi_file(midi_filename)
        # find minimal octave used and tonality of the melody
        self.min_octave, self.key_pair = self.analyze_melody()
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
        for note in self.notes:
            if '#' in note.note:
                sharp_notes += 1
        key_pair = tonic_pairs[sharp_notes]
        return min_octave, key_pair

    @staticmethod
    def get_minor_chord(base: Note) -> Chord:
        return Chord([base, base+3, base+7])

    @staticmethod
    def get_major_chord(base: Note) -> Chord:
        return Chord([base, base+4, base+7])


    def generate_chords(self) -> dict:
        _chords_dict = dict()
        _chords = []
        base = Note('C', 0) # consider all possible notes up to octave the melody itself uses
        while base.octave != self.min_octave:
            minor_chord = Melody.get_minor_chord(base)
            major_chord = Melody.get_major_chord(base)
            # todo add inv, sus
            major_chord_name = str(base)
            minor_chord_name = str(base)+'m'

            major_chords.append(major_chord)
            minor_chords.append(minor_chord)
            _chords_dict[major_chord_name] = major_chord
            _chords_dict[minor_chord_name] = minor_chord
            _chords.append(major_chord)
            _chords.append(minor_chord)

            base = base + 1
        return _chords_dict, _chords


class Note:
    def __init__(self, note: str, octave: int, time_delta=0, midi_code=None):
        assert(note in NOTES)
        self.note = note
        self.octave = octave
        self.time_delta = time_delta
        self.velocity = 50
        self.midi_code = midi_code if midi_code is not None else note_to_number(note, octave)

    @classmethod
    def from_midi(cls, midi_msg: mido.Message):
        assert(midi_msg.type == 'note_off')
        note, octave = number_to_note(midi_msg.note)
        time_delta = midi_msg.time
        midi_code = midi_msg.note
        return cls(note, octave, time_delta, midi_code)

    def __str__(self):
        return f'{self.note}{self.octave}'

    def __repr__(self):
        return f'{self.note}{self.octave}'

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
        return mido.Message(msg_type, note=self.midi_code2, velocity=velocity, time=time)


def number_to_note(number: int) -> tuple:
    octave = number // NOTES_IN_OCTAVE
    note = NOTES[number % NOTES_IN_OCTAVE]
    return note, octave

def note_to_number(note: str, octave: int) -> int:
    assert note in NOTES
    assert octave in OCTAVES
    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)
    assert 0 <= note <= 127
    return note


class Chord:

    def __init__(self, notes: list[Note], rest=False):
        self.rest = rest
        if not rest:
            assert len(notes) > 2
            self.notes = tuple(sorted(notes, key=lambda n: n.midi_code))
            self.name = str(self.notes[0])
            if self.notes[0].distance(self.notes[1]) == 3:
                self.name = self.name + 'm'
        else:
            self.notes = tuple()
            self.name = '<pause>'

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
        return note in self.notes


chords_dict = dict()
minor_chords = []
major_chords = []




def accompany(messages: list[mido.Message], chords: list[Chord]) -> list[mido.Message]:
    i = 0
    new_messages = []
    for msg in messages:
        new_messages.append(msg)
        if msg.type not in ('note_off', 'note_on'):
            continue
        note_toggle = True if msg.type == 'note_on' else False
        chord = chords[i]
        if not chord.rest:
            chord_messages = chord.to_midi_messages(note_on=note_toggle, time=0, velocity=45)
            for note_to_msg in chord_messages:
                new_messages.append(note_to_msg)
        if msg.type == 'note_off':
            i += 1
    return new_messages


def evolution(melody: Melody, generations=50, population_size=70, n_offsprings=25) -> Accompaniment:
    population = initial_population(population_size, melody)
    most_fit_individual = None
    for i in range(generations):
        population = evolution_step(population, n_offsprings)
        most_fit_individual = population[-1]

    return most_fit_individual


def replace_population(population: list[Accompaniment], new_individuals: list[Accompaniment]):
    population_size = len(population)
    population.extend(new_individuals)
    population.sort(key=lambda x: individual_fitness(x))
    return population[population_size:]


def evolution_step(population: list[Accompaniment], n_offsprings: int):
    mothers, fathers = split_population(population, n_offsprings)
    offsprings = []
    for mother, father in zip(mothers, fathers):
        child = mutate(crossover(mother, father))
        offsprings.append(child)
    new_generation = replace_population(population, offsprings)
    return new_generation


def split_population(population: list[Accompaniment], n_offsprings: int) -> tuple:
    mothers = population[-2 * n_offsprings::2]
    fathers = population[-2 * n_offsprings+1::2]
    return mothers, fathers


def crossover(mother: Accompaniment, father: Accompaniment) -> Accompaniment:
    mid = len(mother.chords) // 2
    assert len(mother.chords) == len(father.chords)
    mother_head = mother.chords[:mid].copy()
    mother_tail = mother.chords[mid:].copy()
    father_tail = father.chords[mid:].copy()

    mapping = {father_tail[i]: mother_tail[i] for i in range(len(mother_tail))}

    for i in range(len(mother_head)):
        while mother_head[i] in father_tail:
            mother_head[i] = mapping[mother_head[i]]

    mother_head.extend(father_tail)
    return Accompaniment(mother_head)


def mutate(child: Accompaniment) -> Accompaniment:
    # swap mutation
    i, j = np.random.choice(len(child.chords), 2, replace=False)
    child.chords[i], child.chords[j] = child.chords[j], child.chords[i]
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
    BAD_CHORD_PENALTY = 1000
    LARGE_DIST_PENALTY = 10
    penalty = 0
    max_dist = 1
    for i in range(len(acc.chords)):
        if not acc.chords[i].has_note(melody.notes[i]): # idea 1: chords that don't contain accompanying notes are very bad
            penalty -= BAD_CHORD_PENALTY
        if acc.chords[i] not in good_chords(melody.key_pair[0]):
            penalty -= BAD_CHORD_PENALTY
        if i != len(acc.chords) - 1:
            dist = acc.chords[i].distance(acc.chords[i+1])
            if dist > max_dist:
                max_dist = dist
    penalty -= max_dist*LARGE_DIST_PENALTY # idea 2: distance between chords are undesirable
    return penalty


def get_notes(midi_track) -> list:
    notes = []
    for msg in midi_track:
        if msg.type == 'note_off':
            notes.append(Note.from_midi(msg))
    return notes


def load_midi_file(filename: str):
    mid = mido.MidiFile()
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
    chord = get_major_chord(note)
    #print(chord)
    #print(chord.to_midi_messages(note_on=True))


def do_accompaniment(melody: Melody):

    _chords = []
    for note in melody.notes:
        _chords.append(Melody.get_major_chord(note-36))
    new_messages = accompany(melody.messages, _chords)
    melody.midi_file.tracks[1] = new_messages
    print_midi_file(melody.midi_file)
    melody.midi_file.save('output.mid')


def get_good_chords(tonic):
    pass


melody = Melody()

def main():
    test()
    global melody
    melody = Melody('barbiegirl_mono.mid')
    do_accompaniment(melody)
    print('done')

if __name__ == '__main__':
    main()