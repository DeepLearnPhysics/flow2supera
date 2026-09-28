#!/usr/bin/env python3
'''
Regression test comparing the contents of two flow2supera larcv output files.

Intended use: before committing, run the current code and the previous
commit (or develop) on the same input file, then compare:

    python3 test/test_regression.py reference.root new.root

Trees present in both files must have identical contents entry by entry.
Trees present in only one file are reported (new data products are allowed,
but flagged so the difference is a conscious choice).

Exits non-zero if any common tree differs.
'''

import sys
import numpy as np
import ROOT
from larcv import larcv


def product_and_producer(tree_name):
    # tree names look like <product>_<producer>_tree
    assert tree_name.endswith('_tree')
    product, producer = tree_name[:-len('_tree')].split('_', 1)
    return product, producer


def sparse3d_content(ev):
    return [(v.id(), v.value()) for v in ev.as_vector()]


def cluster3d_content(ev):
    return [[(v.id(), v.value()) for v in c.as_vector()] for c in ev.as_vector()]


def particle_content(ev):
    out = []
    for p in ev.as_vector():
        out.append((p.id(), p.track_id(), p.pdg_code(), p.parent_track_id(),
                    p.parent_pdg_code(), p.ancestor_track_id(), p.ancestor_pdg_code(),
                    p.interaction_id(), p.group_id(), int(p.shape()),
                    p.creation_process(),
                    p.energy_init(), p.energy_deposit(), p.num_voxels(),
                    (p.x(), p.y(), p.z(), p.t()),
                    (p.px(), p.py(), p.pz()),
                    (p.end_position().x(), p.end_position().y(), p.end_position().z()),
                    tuple(p.children_id())))
    return out


def neutrino_content(ev):
    return [(n.id(), n.interaction_id(), n.pdg_code(), n.lepton_pdg_code(),
             n.current_type(), n.interaction_mode(), n.interaction_type(), n.target(),
             n.energy_init(), n.theta(), n.bjorken_x(), n.inelasticity(),
             n.momentum_transfer(), n.momentum_transfer_mag(), n.energy_transfer(),
             n.lepton_p(), (n.x(), n.y(), n.z(), n.t()), (n.px(), n.py(), n.pz()))
            for n in ev.as_vector()]


def opflash_content(ev):
    return [(f.id(), f.time(), f.timeWidth(), f.volume_id(), tuple(f.PEPerOpDet()))
            for f in ev.as_vector()]


def trigger_content(ev):
    return (ev.id(), ev.time_s(), ev.time_ns(), ev.type())


CONTENT_DUMPERS = {
    'sparse3d': sparse3d_content,
    'cluster3d': cluster3d_content,
    'particle': particle_content,
    'neutrino': neutrino_content,
    'opflash': opflash_content,
    'trigger': trigger_content,
}


def tree_content(fname, tree_name):
    f = ROOT.TFile.Open(fname)
    tree = f.Get(tree_name)
    product, producer = product_and_producer(tree_name)
    dumper = CONTENT_DUMPERS[product]
    branch_name = tree_name.replace('_tree', '_branch')
    contents = []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        ev = getattr(tree, branch_name)
        contents.append((dumper(ev), (ev.run(), ev.subrun(), ev.event())))
    f.Close()
    return contents


def list_trees(fname):
    f = ROOT.TFile.Open(fname)
    names = sorted(k.GetName() for k in f.GetListOfKeys() if k.GetName().endswith('_tree'))
    f.Close()
    return names


def main(ref_file, new_file):
    ref_trees = list_trees(ref_file)
    new_trees = list_trees(new_file)

    only_ref = sorted(set(ref_trees) - set(new_trees))
    only_new = sorted(set(new_trees) - set(ref_trees))
    common = sorted(set(ref_trees) & set(new_trees))

    if only_ref:
        print(f'[test_regression] WARNING trees only in {ref_file}: {only_ref}')
    if only_new:
        print(f'[test_regression] INFO new trees in {new_file}: {only_new}')

    n_fail = 0
    for tree_name in common:
        ref = tree_content(ref_file, tree_name)
        new = tree_content(new_file, tree_name)
        if len(ref) != len(new):
            print(f'FAIL {tree_name}: number of entries differ ({len(ref)} vs {len(new)})')
            n_fail += 1
            continue
        bad_entries = [i for i, (r, n) in enumerate(zip(ref, new)) if r != n]
        if bad_entries:
            print(f'FAIL {tree_name}: entries differ at {bad_entries}')
            n_fail += 1
        else:
            print(f'PASS {tree_name} ({len(ref)} entries)')

    if n_fail:
        print(f'\n[test_regression] FAILED: {n_fail} tree(s) differ')
        return 1
    print(f'\n[test_regression] SUCCESS: all {len(common)} common trees identical')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
