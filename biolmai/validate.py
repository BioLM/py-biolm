import re

UNAMBIGUOUS_AA = ('A', 'C', 'D', 'E', 'F', 'G', 'H',
                  'I', 'K', 'L', 'M', 'N', 'P', 'Q',
                  'R', 'S', 'T', 'V', 'W', 'Y' )
AAs = ''.join(UNAMBIGUOUS_AA)
# Let's use extended list for ESM-1v
AAs_EXTENDED = 'ACDEFGHIKLMNPQRSTVWYBXZJUO'


UNAMBIGUOUS_DNA = ('A', 'C', 'T', 'G')
AMBIGUOUS_DNA = ('A', 'C', 'T', 'G', 'X', 'N', 'U')


regexes = {
    'empty_or_unambiguous_aa_validator': re.compile(f"^[{AAs}]*$"),
    'empty_or_unambiguous_dna_validator': re.compile(r"^[ACGT]*$"),
    'extended_aa_validator': re.compile(f"^[{AAs_EXTENDED}]+$"),
    'unambiguous_aa_validator': re.compile(f"^[{AAs}]+$"),
    'unambiguous_dna_validator': re.compile(r"^[ACGT]+$"),
}


def empty_or_unambiguous_aa_validator(txt):
    r = regexes['empty_or_unambiguous_aa_validator']
    if not bool(r.match(txt)):
        err = f"Residues can only be represented with '{AAs}' characters"
        raise AssertionError(err)
    return txt


def empty_or_unambiguous_dna_validator(txt):
    r = regexes['empty_or_unambiguous_dna_validator']
    if not bool(r.match(txt)):
        err = f"Nucleotides can only be represented with 'ACTG' characters"
        raise AssertionError(err)
    return txt


def extended_aa_validator(txt):
    r = regexes['extended_aa_validator']
    if not bool(r.match(txt)):
        err = f"Extended residues can only be represented with " \
              f"'{AAs_EXTENDED}' characters"
        raise AssertionError(err)
    return txt


def unambiguous_aa_validator(txt):
    r = regexes['unambiguous_aa_validator']
    if not bool(r.match(txt)):
        err = f"Unambiguous residues can only be represented with '{AAs}' " \
              f"characters"
        raise AssertionError(err)
    return txt


def unambiguous_dna_validator(txt):
    r = regexes['unambiguous_dna_validator']
    if not bool(r.match(txt)):
        err = f"Unambiguous nucleotides can only be represented with 'ACTG' " \
              f"characters"
        raise AssertionError(err)
    return txt


class UnambiguousAA:
    def __call__(self, value):
        _ = unambiguous_aa_validator(value)


class UnambiguousAAPlusExtra:
    def __init__(self, extra=[]):
        self.extra = extra
        assert len(extra) > 0
        assert isinstance(extra, list)

    def __call__(self, value):
        txt_clean = value
        for ex in self.extra:
            txt_clean = value.replace(ex, '')
        _ = unambiguous_aa_validator(txt_clean)


class ExtendedAAPlusExtra:
    def __init__(self, extra=[]):
        self.extra = extra
        assert len(extra) > 0
        assert isinstance(extra, list)

    def __call__(self, value):
        txt_clean = value
        for ex in self.extra:
            txt_clean = value.replace(ex, '')
        _ = extended_aa_validator(txt_clean)


class SingleOccurrenceOf:
    def __init__(self, single_char):
        self.single_char = single_char

    def __call__(self, value):
        s = self.single_char
        cc = value.count(s)
        if cc != 1:
            err = "Expected a single occurrence of '{}', got {}"
            raise AssertionError(err.format(s, cc))
