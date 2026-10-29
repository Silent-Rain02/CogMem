from cogmem.config import CogMemConfig
from cogmem.memory.chunker import DialogChunker
from cogmem.storage.schema import Doc


def split(text, length=30):
    config = CogMemConfig()
    config.chunk.dialog_max_length = length
    return DialogChunker(config).chunk(Doc(id='dialog-example', text=text))


def test_quoted_turn_words_inside_an_utterance_are_not_structural_boundaries():
    text = '[Recording date: 2020-01-01]\n[entry1] Speaker: ' + 'context ' * 12
    text += '\n[entry2] Speaker: The quote is "Turn on the light".\n'
    assert split(text) == [text]


def test_inline_speaker_label_is_content_not_a_new_turn():
    text = 'User: ' + 'context ' * 12 + '\nThe example contains Assistant: as quoted text.\n'
    assert split(text) == [text]


def test_natural_turn_command_at_line_start_is_not_a_numbered_turn_label():
    text = 'User: ' + 'context ' * 12 + '\nTurn on the light before reading.\n'
    assert split(text) == [text]


def test_actual_speaker_boundaries_preserve_all_whitespace_and_crlf():
    text = '  User: ' + 'context ' * 12 + '\r\n\r\nAssistant: response  \r\nUser: final  \r\n'
    chunks = split(text)
    assert len(chunks) >= 2
    assert ''.join(chunks) == text
    assert chunks[0].startswith('  User:') and chunks[-1].endswith('  \r\n')


def test_numbered_turn_boundaries_remain_supported_and_lossless():
    text = 'Turn 1: ' + 'context ' * 12 + '\nTurn 2: response\n'
    chunks = split(text)
    assert len(chunks) == 2
    assert chunks[1].startswith('Turn 2:')
    assert ''.join(chunks) == text
