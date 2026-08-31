"""Parity-specific myperm definitions for Rubik's cubes."""

from cube.myperm_registry import moves_available_for_size


def bases(cube):
    """Return the size-filtered base sequences shared by parity algorithms."""
    return {
        'swap': moves_available_for_size(cube, ('2F2', '3F2', ' R2', ' U2', '2F2', '3F2', ' U2', ' R2', '2F2', '3F2')),
        'cycle_u': moves_available_for_size(cube, ('2U2', '3U2', ' R2', ' F2', '2U2', '3U2', ' F2', ' R2', '2U2', '3U2')),
        'cycle_d': moves_available_for_size(cube, ('2D2', '3D2', ' L2', ' F2', '2D2', '3D2', ' F2', ' L2', '2D2', '3D2')),
    }


def basic_swap_moves(pll_parity):
    """Return the primary A/B/F/J/K parity-swap move families."""
    return {
        'ParitySwap-A0-': (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ') + pll_parity,
        'ParitySwap-A1-': (" R'", ' F2', ' D2', " B'", " L'", ' B ', ' D2', " F'", ' R ', " F'") + pll_parity,
        'ParitySwap-A2-': pll_parity + (" R'", ' F2', ' D2', " B'", " L'", ' B ', ' L ', ' D2', " L'", " F'", ' R ', ' F2', " L'", " F'", ' L ', ' F2'),
        'ParitySwap-A3-': pll_parity + (' F2', " R'", ' F2', ' D2', " B'", " L'", ' B ', ' L ', ' D2', " L'", " F'", ' R ', ' F2', " L'", " F'", ' L '),
        'ParitySwap-A4-': pll_parity + (' F2', ' U2', ' F2', ' U2', ' F ', ' R ', " L'", ' U2', " R'", ' L ', " F'", ' B ', ' U ', " B'", ' U ', ' R2', " D'", ' F ', ' D ', ' R2'),
        'ParitySwap-A5-': pll_parity + (' U2', ' F2', ' U2', ' F ', ' R ', " L'", ' U2', " R'", ' L ', " F'", ' B ', ' U ', " B'", ' U ', ' R2', " D'", ' F ', ' D ', ' R2', " F2"),
        'ParitySwap-B0-': pll_parity + (" L2", " F2", " U2", " L'", " U2", " L2", " F2", " L'", " U2", " L2", " U2", " F2", " L'", " F2"),
        'ParitySwap-B1-': pll_parity + (" F2", " L ", " F2", " U2", " L2", " U2", " L ", " F2", " L2", " U2", " L ", " U2", " F2", " L2"),
        'ParitySwap-B2-': (" R2", " U2", " B2", " R'", " B2", " R2", " U2", " R ", " B2", " R2", " B2", " U2", " R ", " U2") + pll_parity,
        'ParitySwap-B3-': (" U2", " R'", " U2", " B2", " R2", " B2", " R'", " U2", " R2", " B2", " R ", " B2", " U2", " R2") + pll_parity,
        'ParitySwap-F0-': pll_parity + (" R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ', " F "),
        'ParitySwap-F1-': pll_parity + (" F'", " R'", " F2", " D2", " B'", " L'", " B ", " D2", " F'", " R "),
        'ParitySwap-F2-': (" B ", " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', " B2") + pll_parity,
        'ParitySwap-F3-': (' B2', " L'", ' B ', ' D2', " F'", ' R ', ' F ', ' D2', ' B2', ' L ', " B'") + pll_parity,
        'ParitySwap-F4-': (" U2", " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2', ' R ', " F ", " U2") + pll_parity,
        'ParitySwap-F5-': (" U2", " F'", " R'", " F2", " D2", " B'", " L'", " B ", " D2", " F'", " R ", " U2") + pll_parity,
        'ParitySwap-J0-': (' B2', " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', ' B ') + pll_parity,
        'ParitySwap-J1-': (" B'", " L'", ' B ', ' D2', " F'", ' R ', ' F ', ' D2', ' B2', ' L ', ' B2') + pll_parity,
        'ParitySwap-J2-': (' F2', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ', " L'", ' F ', ' R ', " F'", ' L ') + pll_parity,
        'ParitySwap-J3-': (" L'", ' F ', " R'", " F'", ' L ', " F'", ' D2', " B'", " L'", ' B ', ' D2', " F'", ' R ', ' F2') + pll_parity,
        'ParitySwap-J4-': (' B2', ' L2', ' D2', ' F2', ' D2', ' L2', " B'", ' U2', ' L2', ' D ', ' F ', " D'", ' L2', ' U ', " B'", " U'") + pll_parity,
        'ParitySwap-J5-': (' U ', ' B ', " U'", ' L2', ' D ', " F'", " D'", ' L2', ' U2', ' B ', ' L2', ' D2', ' F2', ' D2', ' L2', ' B2') + pll_parity,
        'ParitySwap-K0-': (" R'", ' U2', ' L ', ' F2', " L'", ' F2', ' R2', ' U2', ' R ', ' U2', " R'", ' U2', ' F2', ' R2', ' F2') + pll_parity,
        'ParitySwap-K1-': (' R2', ' F2', ' U2', ' R ', ' U2', " R'", ' U2', ' R2', ' F2', ' L ', ' F2', " L'", ' U2', ' R ', ' F2') + pll_parity,
    }


def register_cycle_algorithms(cube, parity_bases):
    """Register parity-cycle algorithms and odd-cube center permutations."""
    cycle_u = parity_bases['cycle_u']
    cycle_d = parity_bases['cycle_d']
    algorithms = (
        ('C4[DFR>FUR>LFD>LUF]+ME2[FL>FR]', (" U'", " R'", " F'", ' R ', ' F ', " R'", ' F ', ' R ', ' F ', " R'", " F'", ' R ', ' U ') + cycle_u),
        ('C4[DFR>LUF>LFD>FUR]+ME2[FL>FR]', (" U'", " R'", ' F ', ' R ', " F'", " R'", " F'", ' R ', " F'", " R'", ' F ', ' R ', ' U ') + cycle_u),
        ('C4[DBL>FLU>DRB>FUR]+ME2[FL>FR]', (' D ', " L'", " F'", ' L ', ' F ', " L'", ' F ', ' L ', ' F ', " L'", " F'", ' L ', " D'") + cycle_d),
        ('C4[DBL>FUR>DRB>FLU]+ME2[FL>FR]', (' D ', " L'", ' F ', ' L ', " F'", " L'", " F'", ' L ', " F'", " L'", ' F ', ' L ', " D'") + cycle_d),
        ('C4[DBL>FUR>DLF>FLU]+ME2[FL>FR]', (" L'", " F'", ' L ', ' F ', " L'", ' F ', ' L ', ' F ', " L'", " F'", ' L ') + cycle_d),
        ('C4[DBL>FLU>DLF>FUR]+ME2[FL>FR]', (" L'", ' F ', ' L ', " F'", " L'", " F'", ' L ', " F'", " L'", ' F ', ' L ') + cycle_d),
    )
    for name, moves in algorithms:
        cube._add_myperm2(name, moves)
    if cube.size % 2 == 1:
        cube._add_myperm2('CtrCore6p[3x2][B>R>D;F>L>U]', (" M ", " E ", " M'", " E'"))
        cube._add_myperm2('CtrCore4s[B<>F;L<>R]', (' E ', ' S2', " E'", ' S2'))


def add_x_prefix_swaps(parity_swap_moves, pll_parity):
    """Add the directly specified X-family parity swaps."""
    parity_swap_moves['ParitySwap-XB-'] = (' U ', " F'", ' R ') + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F'", " U'")
    parity_swap_moves['ParitySwap-XC-'] = (' F2', ' R ') + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2')
    parity_swap_moves['ParitySwap-XD-'] = (' F ', ' R ') + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ')
    parity_swap_moves['ParitySwap-XE-'] = (' R ',) + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F2')
    parity_swap_moves['ParitySwap-XF-'] = (" F'", ' R ') + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F'")


def register_super_swaps(cube, pll_parity):
    """Register the smaller-cube SuperParitySwap variants."""
    cube.myperms2['SuperParitySwap-JC00-'] = (" D2", ' F2', ' R ') + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ')
    cube.myperms2['SuperParitySwap-JE00-'] = (" U2", ' F2', ' R ') + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', " D2", " U2")
    cube.myperms2['SuperParitySwap-JD00-'] = (" R2", ' F ', ' R ') + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ', " R2")
    cube.myperms2['SuperParitySwap-JF00-'] = (" L2", ' F ', ' R ') + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ', " L2")
    cube.myperms2['SuperParitySwap-JC01-'] = cube.conjugate((" z'", " F ", " B'", " y ", " U'", " D "), cube.myperms2['SuperParitySwap-JC00-'])
    cube.myperms2['SuperParitySwap-JD01-'] = cube.conjugate((" z ", " F'", " B ", " x ", " L ", " R'"), cube.myperms2['SuperParitySwap-JD00-'])
    cube.myperms2['SuperParitySwap-JE01-'] = cube.conjugate((" z ", " F'", " B ", " y ", " U'", " D "), cube.myperms2['SuperParitySwap-JE00-'])
    cube.myperms2['SuperParitySwap-JF01-'] = cube.conjugate((" z'", " F ", " B'", " x ", " L ", " R'"), cube.myperms2['SuperParitySwap-JF00-'])


def register_size_filtered(cube, name, moves):
    """Register a parity algorithm after removing unavailable inner layers."""
    cube._add_myperm2(name, cube._moves_available_for_size(moves))


def add_derived_swaps(cube, parity_swap_moves, pll_parity):
    """Add conjugated X/Y/Z and composite parity-swap families."""
    parity_swap_moves['ParitySwap-XG-'] = cube.conjugate((" R2",), parity_swap_moves['ParitySwap-A0-'])
    parity_swap_moves['ParitySwap-XH-'] = cube.conjugate((" U'", " F'", " R "), parity_swap_moves['ParitySwap-A0-'])
    parity_swap_moves['ParitySwap-YA-'] = pll_parity + (" R ", " U ", " R'", " U'", " R'", " F ", " R2", " U'", " R'", " U'", " R ", " U ", " R'", " F'")
    for suffix, conjugator in (('YB', (" U ", " F'", " R ")), ('YC', (" F2", " R ")), ('YD', (" F ", " R ")), ('YE', (" R ",)), ('YF', (" F'", " R ")), ('YG', (" R2",)), ('YH', (" R'", " U'", " F ", " U "))):
        parity_swap_moves['ParitySwap-' + suffix + '-'] = cube.conjugate(conjugator, parity_swap_moves['ParitySwap-YA-'])
    parity_swap_moves['ParitySwap-ZA-'] = pll_parity + (' U2', " B'", ' U2', ' B ', ' U2', ' D2', " R'", " B'", ' R ', ' D2', " L'", ' F ', " L'", " F'", ' L2')
    for suffix, conjugator in (('ZB', (" F'", " U ", " L'")), ('ZC', (" U2", " L ")), ('ZD', (" U ", " L ")), ('ZE', (" L ",)), ('ZF', (" U'", " L ")), ('ZG', (" L2",)), ('ZH', (" F ", " U ", " L'"))):
        parity_swap_moves['ParitySwap-' + suffix + '-'] = cube.conjugate(conjugator, parity_swap_moves['ParitySwap-ZA-'])
    parity_swap_moves['ParitySwap-JXB-'] = (" R2", ' U ', " F'", ' R ') + pll_parity + (' F ', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', " F'", " U'", " R2")
    parity_swap_moves['ParitySwap-JYB-'] = (" R2", ' U ', " F'", ' R ') + pll_parity + (' R ', ' U ', " R'", " U'", " R'", ' F ', ' R2', " U'", " R'", " U'", ' R ', ' U ', " R'", " F'", " R'", ' F ', " U'", " R2")
    parity_swap_moves['ParitySwap-JZB-'] = (" U'", ' B ', " R'") + pll_parity + (" B'", ' R ', " B'", ' D2', ' F ', " L'", " F'", ' D2', ' B ', ' U ', ' L ', ' U2', " L'", ' D ', ' L ', ' U2', " L'", " D'")
