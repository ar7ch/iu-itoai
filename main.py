import numpy
import mido

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

class Note:
    def __init__(self, note, octave):
        assert note in NOTES
        self.note = note
        self.octave = octave

    def __str__(self):
        return f'{self.note}{self.octave}'

    def __repr__(self):
        return f'{self.note}{self.octave}'

class Chord:
    def __init__(self, notes):
        self.notes = notes

    def __str__(self):
        return f'{self.notes}'

    def __repr__(self):
        return f'{self.notes}'


def number_to_note(number: int) -> Note:
    octave = number // NOTES_IN_OCTAVE
    note = NOTES[number % NOTES_IN_OCTAVE]
    return Note(note, octave)

def get_notes(midi_track) -> list:
    notes = []
    for msg in midi_track:
        if msg.type == 'note_on':
            notes.append(number_to_note(msg.note))
    return notes

def load_midi():
    mid = mido.MidiFile('barbiegirl_mono.mid')
    notes = get_notes(mid.tracks[1])
    print(notes)

def main():
    load_midi()


if __name__ == '__main__':
    main()