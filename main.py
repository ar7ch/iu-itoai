import numpy
import mido

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

class Note:
    def __init__(self, note, octave, duration=None, midi_code=None):
        assert(note in NOTES)
        self.note = note
        self.octave = octave
        self.duration = duration
        self.midi_code = midi_code if midi_code is not None else note_to_number(note, octave)

    @classmethod
    def from_midi(cls, midi_msg):
        assert(midi_msg.type == 'note_off')
        note, octave = number_to_note(midi_msg.note)
        duration = midi_msg.time
        midi_code = midi_msg.note
        return cls(note, octave, duration, midi_code)

    def __str__(self):
        return f'{self.note}{self.octave}'

    def __repr__(self):
        return f'{self.note}{self.octave}'

    def __add__(self, other: int):
        code = self.midi_code + other
        note, octave = number_to_note(code)
        return Note(note, octave, self.duration, code)

    def __sub__(self, other: int):
        return self + -other


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
    def __init__(self, notes):
        self.notes = notes

    def __str__(self):
        return f'{self.notes}'

    def __repr__(self):
        return f'{self.notes}'

def generate_chords(base_octave: int):
    pass


def get_notes(midi_track) -> list:
    notes = []
    for msg in midi_track:
        if msg.type == 'note_off':
            notes.append(Note.from_midi(msg))
    return notes


def load_midi():
    mid = mido.MidiFile('barbiegirl_mono.mid')
    for tr in mid.tracks[1]:
        print(tr)
    #notes = get_notes(mid.tracks[1])
    #print(notes)

def test():
    note = Note('C', 3)
    print(note - 1)

def main():
    test()
    #load_midi()

if __name__ == '__main__':
    main()