import matplotlib.pyplot as plt


def expand_chunks(chunks_grid):
    """
    [["AA","B"], ["AA","C"], ["D","EE"]] ->
    [["A","A","B"], ["A","A","C"], ["D","E","E"]]
    """
    grid = []
    for row in chunks_grid:
        out = []
        for chunk in row:
            if not isinstance(chunk, str):
                raise TypeError(f"chunk must be str, got {type(chunk)}: {chunk!r}")
            out.extend(list(chunk))
        grid.append(out)

    widths = {len(r) for r in grid}
    if len(widths) != 1:
        raise ValueError(f"expanded rows have different widths: {sorted(widths)}")
    return grid


def mosaic_to_subfigures(fig, subplot_spec, chunks_grid, *, empty_sentinel="."):
    """
    Create SubFigure objects (not Axes) laid out like a mosaic.

    Returns: dict[label] -> SubFigure

    fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    outer = fig.add_gridspec(1, 1)

    chunks = [["AA", "B"],
              ["AA", "C"],
              ["D",  "EE"]]

    subfigs = mosaic_to_subfigures(fig, outer[0, 0], chunks)

    # Each subfigure can now contain its own axes
    for lab, sf in subfigs.items():
        sf.suptitle(lab, x=0.02, ha="left")  # subfigure title
        ax = sf.add_subplot(111)
        ax.text(0.5, 0.5, f"axes inside subfig {lab}", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
    """
    grid = expand_chunks(chunks_grid)
    nrows, ncols = len(grid), len(grid[0])

    # Build a sub-gridspec covering the whole region we were given
    gs = subplot_spec.subgridspec(nrows=nrows, ncols=ncols)

    # Collect bounding boxes per label
    boxes = {}  # label -> [rmin, rmax, cmin, cmax]
    for r in range(nrows):
        for c in range(ncols):
            lab = grid[r][c]
            if lab == empty_sentinel:
                continue
            if lab not in boxes:
                boxes[lab] = [r, r, c, c]
            else:
                boxes[lab][0] = min(boxes[lab][0], r)
                boxes[lab][1] = max(boxes[lab][1], r)
                boxes[lab][2] = min(boxes[lab][2], c)
                boxes[lab][3] = max(boxes[lab][3], c)

    # Validate: each label must occupy a full rectangle (no holes / non-rect shapes)
    for lab, (r0, r1, c0, c1) in boxes.items():
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if grid[rr][cc] != lab:
                    raise ValueError(
                        f"Label {lab!r} is not rectangular (hole or L-shape) "
                        f"at row={rr}, col={cc}."
                    )

    # Create subfigures
    subfigs = {}
    for lab, (r0, r1, c0, c1) in boxes.items():
        spec = gs[r0:r1 + 1, c0:c1 + 1]
        subfigs[lab] = fig.add_subfigure(spec)

    return subfigs

