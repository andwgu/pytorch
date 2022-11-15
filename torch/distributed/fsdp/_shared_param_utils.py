import collections

from typing import Dict, List, NamedTuple, Set, Tuple

import torch
import torch.nn as nn


class SharedParamInfo(NamedTuple):
    module1: nn.Module
    module2: nn.Module
    param: nn.Parameter


def get_shared_param_info_to_lca(
    root_module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> Dict[SharedParamInfo, nn.Module]:
    """
    Computes the lowest common ancestors for the queries encoded by
    ``shared_param_infos``, where each entry in the list gives two modules,
    which each represent a vertex in the module tree.
    """
    shared_param_infos = _get_shared_param_infos(root_module, ignored_params)
    # Construct edge list in the LCA query graph
    module_to_sharing_modules: Dict[
        nn.Module, List[nn.Module]
    ] = collections.defaultdict(list)
    for module1, module2, _ in shared_param_infos:
        module_to_sharing_modules[module1].append(module2)
        module_to_sharing_modules[module2].append(module1)

    parent: Dict[nn.Module, nn.Module] = {}
    rank: Dict[nn.Module, int] = {}
    ancestor: Dict[nn.Module, nn.Module] = {}
    color: Dict[nn.Module, str] = {}
    lca_query_to_lca: Dict[Tuple[nn.Module, nn.Module], nn.Module] = {}

    def tarjan_lca(module: nn.Module):
        make_set(module, parent, rank)
        for child_module in module.children():
            tarjan_lca(child_module)
            union(module, child_module, parent, rank)
            ancestor[find(module, parent)] = module
        color[module] = "black"
        if module not in module_to_sharing_modules:
            return
        for other_module in module_to_sharing_modules[module]:
            if color.get(other_module, "") == "black":
                lca_query = _get_lca_query(module, other_module)
                if lca_query in lca_query_to_lca:
                    continue  # already computed
                lca_query_to_lca[lca_query] = ancestor[find(other_module, parent)]

    tarjan_lca(root_module)  # compute the LCAs
    shared_param_info_to_lca: Dict[SharedParamInfo, nn.Module] = {}
    for shared_param_info in shared_param_infos:
        module1, module2, _ = shared_param_info
        lca_query = _get_lca_query(module1, module2)
        assert lca_query in lca_query_to_lca, f"Missing LCA query: {lca_query}"
        shared_param_info_to_lca[shared_param_info] = lca_query_to_lca[lca_query]
    return shared_param_info_to_lca


def _get_shared_param_infos(
    root_module: nn.Module,
    ignored_params: Set[nn.Parameter],
) -> List[SharedParamInfo]:
    visited_param_to_module: Dict[nn.Parameter, nn.Module] = {}
    shared_param_infos: List[SharedParamInfo] = []
    for module in root_module.modules():
        for param in module.parameters(recurse=False):
            if param in ignored_params:
                continue
            if param in visited_param_to_module:  # shared parameter
                shared_param_infos.append(
                    SharedParamInfo(visited_param_to_module[param], module, param)
                )
            else:
                visited_param_to_module[param] = module
    return shared_param_infos


def _get_lca_query(
    module1: nn.Module, module2: nn.Module
) -> Tuple[nn.Module, nn.Module]:
    return (module1, module2) if id(module1) < id(module2) else (module2, module1)


def make_set(module, parent, rank) -> None:
    assert isinstance(module, nn.Module)
    parent[module] = module
    rank[module] = 1


def union(module1, module2, parent, rank) -> None:
    assert isinstance(module1, nn.Module)
    assert isinstance(module2, nn.Module)
    root1 = find(module1, parent)
    root2 = find(module2, parent)
    if rank[root1] > rank[root2]:
        parent[root2] = root1
    elif rank[root1] < rank[root2]:
        parent[root1] = root2
    else:
        parent[root2] = root1
        rank[root1] += 1


def find(module, parent):
    assert isinstance(module, nn.Module)
    if parent[module] != module:
        parent[module] = find(parent[module], parent)
    return parent[module]
