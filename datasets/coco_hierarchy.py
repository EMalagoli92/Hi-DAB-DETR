import json
from pathlib import Path


def get_coco_wordnet_hierarchy(
        tree_path: str | Path,
        names_path: str | Path,
        coco_map_path: str | Path,
        ) -> tuple[dict[int, int], dict[int, str], list[int]]:
    """
    Extract the COCO subtree from the WordTree.

    Parameters
    ----------
    tree_path : str | Path
        Path to '9k.tree.txt' (WordTree structure).
    names_path : str | Path
        Path to '9k.names.txt' (WordTree names).
    coco_map_path : str | Path
        Path to 'coco9k.map.txt' (COCO node indices).

    Returns
    -------
    dict[int, int]
        WordTree subtree with COCO and internal nodes (child -> parent).
    dict[int, str]
        Node names for the subtree.
    list[int]
        WordTree node IDs for COCO classes.
    """
    # WordTree: <synset> <parent>
    with Path(tree_path).open() as handle:
        tree = [int(line.strip().split()[1]) for line in handle]
    hierarchy = {idx: tree[idx] for idx in range(len(tree))}

    # Names
    with Path(names_path).open() as handle:
        names = [line.strip() for line in handle]
    if len(names) != len(tree):
        msg = "Mismatch: 9k.names.txt and 9k.tree.txt not aligned"
        raise ValueError(msg)
    hierarchy_names = {idx: names[idx] for idx in range(len(tree))}

    # COCO subtree
    with Path(coco_map_path).open() as handle:
        coco_map = [int(line.strip()) for line in handle]
    coco_subtree_nodes = set()
    for node_id in coco_map:
        while node_id != -1:
            coco_subtree_nodes.add(node_id)
            node_id = hierarchy[node_id]  # noqa: PLW2901
    hierarchy_coco = {k: v for k, v in hierarchy.items()
                      if k in coco_subtree_nodes}
    names_coco = {k: v for k, v in hierarchy_names.items()
                  if k in coco_subtree_nodes}

    return hierarchy_coco, names_coco, coco_map

def remap_coco_hierarchy(
        coco_categories_path: str | Path,
        hierarchy_coco: dict[int, int],
        names_coco: dict[int, str],
        coco_map: list[int]
    ) -> tuple[dict[int, int], dict[int, str], list[int]]:
    """
    Remap COCO WordTree nodes.

    Parameters
    ----------
    coco_categories_path : str | Path
        Path to COCO category ID file (e.g. 'coco_categories.txt').
    hierarchy_coco : dict[int, int]
        Subtree hierarchy from WordTree.
    names_coco : dict[int, str]
        Node names for the subtree.
    coco_map : list[int]
        WordTree node IDs for COCO classes.

    Returns
    -------
    dict[int, int]
        Remapped hierarchy.
    dict[int, str]
        Remapped node names.
    list[int]
        Remapped model indices for COCO classes.
    dict[int, str]
        COCO category ID to original COCO name.
    """
    with Path(coco_categories_path).open() as handle:
        coco_categories = [line.strip() for line in handle]
    original_coco_names = {int(line.split()[0]): " ".join(line.split()[1:])
                           for line in coco_categories}
    coco_categories_map = {idx: int(coco_categories[idx].split()[0])
                           for idx in range(len(coco_categories))}
    wordtree_to_category_id = {
        coco_map[i]: coco_categories_map[i] for i in range(len(coco_map))
    }
    other_nodes = [node_id for node_id in hierarchy_coco
                   if node_id not in wordtree_to_category_id]
    # Map non-COCO nodes starting from 92
    # (0-90 reserved for COCO, 91 = no-object)
    start_other_index = 92
    wordtree_to_model_idx = wordtree_to_category_id | {
        node_id: start_other_index + i for i, node_id in enumerate(other_nodes)
    }
    hierarchy_remapped = {
        wordtree_to_model_idx[child]: (
            wordtree_to_model_idx[parent] if parent != -1 else -1)
        for child, parent in hierarchy_coco.items()
    }

    # Fill missing in 0-91: COCO holes (0-90) + no-object (91)
    # with root (-1)
    missing = [node_id for node_id in range(start_other_index)
               if node_id not in list(hierarchy_remapped)]
    hierarchy_remapped |= dict.fromkeys(missing, -1)

    # Sort
    hierarchy_remapped = dict(sorted(hierarchy_remapped.items()))

    # Remap names
    names_remapped = {
        wordtree_to_model_idx[node_id]: names_coco[node_id]
        for node_id in wordtree_to_model_idx
    }
    # Add missing node names: assign "no-object" to 91, "missing_{id}"
    # to others
    for node_id in missing:
        if node_id == 91:  # noqa: PLR2004
            names_remapped[node_id] = "no-object"
        else:
            names_remapped[node_id] = f"missing_{node_id}"
    # Sort
    names_remapped = dict(sorted(names_remapped.items()))

    coco_map_remapped = [wordtree_to_model_idx[node_id]
                         for node_id in coco_map]

    return (hierarchy_remapped, names_remapped, coco_map_remapped,
            original_coco_names)


def build_wordtree_coco_hierarchy(
        word_tree_dir: str | Path
    ) -> tuple[dict[int, int], dict[int, str], list[int]]:
    """
    Build remapped COCO hierarchy from WordTree files.

    Parameters
    ----------
    word_tree_dir : str | Path
        Directory containing WordTree and COCO mapping files.

    Returns
    -------
    dict[int, int]
        Remapped hierarchy.
    dict[int, str]
        Remapped node names.
    list[int]
        Remapped model indices for COCO classes.
    """
    word_tree_dir = Path(word_tree_dir)

    hierarchy_coco, names_coco, coco_map = get_coco_wordnet_hierarchy(
        tree_path=word_tree_dir / "9k.tree.txt",
        names_path=word_tree_dir / "9k.names.txt",
        coco_map_path=word_tree_dir / "coco9k.map.txt"
    )

    return remap_coco_hierarchy(
        coco_categories_path=(word_tree_dir / "coco_categories.txt"),
        hierarchy_coco=hierarchy_coco,
        names_coco=names_coco,
        coco_map=coco_map
    )


def create_coco_categories(
        annotation_path: str | Path,
        write_path: str | Path
        ) -> None:
    """
    Extract COCO category IDs and names from a JSON annotation file.

    Parameters
    ----------
    annotation_path : str or Path
        Path to COCO JSON file.
    write_path : str or Path
        Destination path for the output text file.
    """
    annotation_path = Path(annotation_path)
    write_path = Path(write_path)

    with annotation_path.open("r") as handle:
        annotations = json.load(handle)

    anns = [(ann_["id"], ann_["name"]) for ann_ in annotations["categories"]]

    with write_path.open("w") as handle:
        for cat_id, name in anns:
            handle.write(f"{cat_id} {name}\n")
