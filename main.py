from __future__ import annotations
import mido

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)


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
        return mido.Message(msg_type, note=self.midi_code+12, velocity=velocity, time=time)


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


chords_dict = dict()
minor_chords = []
major_chords = []


def get_minor_chord(base: Note) -> Chord:
    return Chord([base, base+3, base+7])


def get_major_chord(base: Note) -> Chord:
    return Chord([base, base+4, base+7])

def generate_chords(melody_min_octave: int):
    base = Note('C', 0)
    while base.octave != melody_min_octave:
        minor_chord = get_minor_chord(base)
        major_chord = get_major_chord(base)
        # todo add inv, sus
        major_chord_name = str(base)
        minor_chord_name = str(base)+'m'

        major_chords.append(major_chord)
        minor_chords.append(minor_chord)
        chords_dict[major_chord_name] = major_chord
        chords_dict[minor_chord_name] = minor_chord

        base = base + 1


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
            off_time = 0
            if not note_toggle and msg.time != 384:
                off_time = 384 - msg.time
            # for first note_off message, make offset such that it complements the whole clock
            # then we can make offsets 0"""
            chord_messages = chord.to_midi_messages(note_on=note_toggle, time=0, velocity=49)
            chord_messages[0].time = off_time
            for note_to_msg in chord_messages:
                new_messages.append(note_to_msg)
        if msg.type == 'note_off':
            i += 1
    return new_messages


def evolution(generations: int, population_size: int) -> list[Chord]:
    pass


def get_notes(midi_track) -> list:
    notes = []
    for msg in midi_track:
        if msg.type == 'note_off':
            notes.append(Note.from_midi(msg))
    return notes


def load_midi_file():
    mid = mido.MidiFile('barbiegirl_mono.mid')
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


def do_accompaniment(file, messages, notes):
    _chords = []
    for note in notes:
        _chords.append(get_major_chord(note-24))
    new_messages = accompany(messages, _chords)
    file.tracks[1] = new_messages
    print_midi_file(file)
    file.save('output.mid')

def main():
    test()
    file, messages, notes = load_midi_file()
    do_accompaniment(file, messages, notes)
    print('done')

if __name__ == '__main__':
    main()