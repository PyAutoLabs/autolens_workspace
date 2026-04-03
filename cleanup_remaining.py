#!/usr/bin/env python3
"""Final cleanup pass for remaining old API patterns."""
import re
import sys
from pathlib import Path

# ── Text substitutions for docstrings/comments ──────────────────────────────
# These replace backtick-quoted old names with new API references

DOCSTRING_SUBS = [
    # Plotter class references in docstrings
    (r'`ImagingPlotter`', '`aplt.subplot_imaging_dataset`'),
    (r'`FitImagingPlotter`', '`aplt.subplot_fit_imaging`'),
    (r'`FitInterferometerPlotter`', '`aplt.subplot_fit_interferometer`'),
    (r'`InterferometerPlotter`', '`aplt.subplot_interferometer_dataset`'),
    (r'`TracerPlotter`', '`aplt.subplot_tracer`'),
    (r'`LightProfilePlotter`', '`aplt.plot_array`'),
    (r'`MassProfilePlotter`', '`aplt.plot_array`'),
    (r'`GalaxyPlotter`', '`aplt.plot_array`'),
    (r'`GalaxiesPlotter`', '`aplt.plot_array`'),
    (r'`InversionPlotter`', '`aplt.plot_array`'),
    (r'`MapperPlotter`', '`aplt.plot_array`'),
    (r'`SubhaloPlotter`', '`aplt.subplot_detection_imaging`'),
    (r'`Array2DPlotter`', '`aplt.plot_array`'),
    (r'`Grid2DPlotter`', '`aplt.plot_grid`'),
    (r'`Plotter`', 'plotting function'),
    (r'`PLotter`', 'plotting function'),
    # MatPlot2D/Visuals2D references
    (r'`MatPlot2D`', '`plot_array`/`subplot_\*`'),
    (r'`Visuals2D`', '`lines=`/`positions=` overlays'),
    (r'`MatPlot1D`', '`plot_yx`'),
    # method references in text
    (r'`figures_2d`', '`plot_array`'),
    (r'\.figures_2d\b', '.plot_array'),
    (r'`subplot_fit\(\)`', '`aplt.subplot_fit_imaging`'),
    # phrase substitutions
    (r'use a `aplt\.plot_array`', 'use `aplt.plot_array`'),
]


def apply_docstring_subs(content):
    """Apply text substitutions everywhere (comments, docstrings, code lines)."""
    for pattern, replacement in DOCSTRING_SUBS:
        content = re.sub(pattern, replacement, content)
    return content


def fix_code_patterns(content):
    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # ── figures_2d_of_planes calls ────────────────────────────────────
        m = re.match(
            r'^(\s*)(\w+)\.figures_2d_of_planes\s*\((.+)\)\s*$', line
        )
        if m:
            indent = m.group(1)
            var = m.group(2)
            args_str = m.group(3)

            plane_idx_m = re.search(r'plane_index\s*=\s*(\d+)', args_str)
            plane_idx = plane_idx_m.group(1) if plane_idx_m else '0'

            if 'plane_image=True' in args_str:
                # Determine the fit/tracer variable
                if 'fit_plotter' in var or 'fit' in var:
                    result.append(
                        f"{indent}aplt.plot_array(array=fit.model_images_of_planes_list[{plane_idx}], "
                        f"title=\"Plane {plane_idx} Image\")"
                    )
                else:
                    result.append(
                        f"{indent}aplt.plot_array(array=tracer.image_2d_list_from(grid=grid)[{plane_idx}], "
                        f"title=\"Plane {plane_idx} Image\")"
                    )
            elif 'plane_grid=True' in args_str:
                result.append(
                    f"{indent}aplt.plot_grid(grid=tracer.traced_grid_2d_list_from(grid=grid)[{plane_idx}], "
                    f"title=\"Plane {plane_idx} Grid\")"
                )
            else:
                result.append(
                    f"{indent}aplt.plot_array(array=fit.model_images_of_planes_list[{plane_idx}], "
                    f"title=\"Plane {plane_idx}\")"
                )
            i += 1
            continue

        # ── multi-line figures_2d_of_planes ──────────────────────────────
        m = re.match(r'^(\s*)(\w+)\.figures_2d_of_planes\s*\(', line)
        if m:
            indent = m.group(1)
            # Collect full call
            full = line
            depth = full.count('(') - full.count(')')
            j = i + 1
            while depth > 0 and j < len(lines):
                full += ' ' + lines[j].strip()
                depth += lines[j].count('(') - lines[j].count(')')
                j += 1
            plane_idx_m = re.search(r'plane_index\s*=\s*(\d+)', full)
            plane_idx = plane_idx_m.group(1) if plane_idx_m else '0'
            if 'plane_image=True' in full:
                result.append(
                    f"{indent}aplt.plot_array(array=fit.model_images_of_planes_list[{plane_idx}], "
                    f"title=\"Plane {plane_idx} Image\")"
                )
            elif 'plane_grid=True' in full:
                result.append(
                    f"{indent}aplt.plot_grid(grid=tracer.traced_grid_2d_list_from(grid=grid)[{plane_idx}], "
                    f"title=\"Plane {plane_idx} Grid\")"
                )
            else:
                result.append(
                    f"{indent}aplt.plot_array(array=fit.model_images_of_planes_list[{plane_idx}], "
                    f"title=\"Plane {plane_idx}\")"
                )
            i = j
            continue

        # ── figures_2d_of_pixelization calls ─────────────────────────────
        m = re.match(r'^(\s*)(\w+)\.figures_2d_of_pixelization\s*\(', line)
        if m:
            indent = m.group(1)
            full = line
            depth = full.count('(') - full.count(')')
            j = i + 1
            while depth > 0 and j < len(lines):
                full += ' ' + lines[j].strip()
                depth += lines[j].count('(') - lines[j].count(')')
                j += 1
            if 'reconstruction=True' in full:
                result.append(
                    f"{indent}aplt.plot_array(array=fit.inversion.reconstruction, "
                    f"title=\"Inversion Reconstruction\")"
                )
            else:
                result.append(
                    f"{indent}aplt.plot_array(array=fit.inversion.reconstruction, "
                    f"title=\"Inversion\")"
                )
            i = j
            continue

        # ── Standalone aplt.XPlotter(args) without variable assignment ────
        m = re.match(r'^(\s*)aplt\.(\w+Plotter)\((.+)\)\s*$', line)
        if m:
            indent = m.group(1)
            ptype = m.group(2)
            args = m.group(3)
            # Extract the main array/grid arg
            if ptype == 'Array2DPlotter':
                arr = re.search(r'array\s*=\s*([\w\.]+)', args)
                if arr:
                    result.append(f'{indent}aplt.plot_array(array={arr.group(1)}, title="")')
                    i += 1
                    continue
            elif ptype == 'Grid2DPlotter':
                g = re.search(r'grid\s*=\s*([\w\.]+)', args)
                if g:
                    result.append(f'{indent}aplt.plot_grid(grid={g.group(1)}, title="")')
                    i += 1
                    continue
            # Other standalone plotters without variable — just remove
            i += 1
            continue

        # ── FitPointDatasetPlotter ────────────────────────────────────────
        m_assign = re.match(r'^(\s*)(\w+)\s*=\s*aplt\.FitPointDatasetPlotter\s*\(', line)
        if m_assign:
            indent = m_assign.group(1)
            var = m_assign.group(2)
            full = line
            depth = full.count('(') - full.count(')')
            j = i + 1
            while depth > 0 and j < len(lines):
                full += '\n' + lines[j]
                depth += lines[j].count('(') - lines[j].count(')')
                j += 1
            fit_val = re.search(r'fit\s*=\s*([\w\.]+)', full)
            fit_str = fit_val.group(1) if fit_val else 'fit'
            # Now consume subsequent method call if present
            # We'll store as a temporary note and handle on next line
            result.append(f"# _FitPointDatasetPlotter_{var}_{fit_str}_")
            i = j
            continue

        # Handle stored FitPointDatasetPlotter method calls
        m_fp = re.match(r'^(\s*)(\w+)\.(subplot_fit|figures_2d)\s*\(', line)
        if m_fp:
            indent = m_fp.group(1)
            var = m_fp.group(2)
            # Check if there's a stored note for this var
            for idx, prev in enumerate(result):
                note_m = re.match(rf'^# _FitPointDatasetPlotter_{re.escape(var)}_(\w+)_$', prev)
                if note_m:
                    fit_str = note_m.group(1)
                    result[idx] = f"{indent}aplt.subplot_fit_point(fit={fit_str})"
                    # Consume the method call
                    _, j = _collect_balanced_inline(lines, i)
                    i = j
                    break
            else:
                result.append(line)
                i += 1
            continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def _collect_balanced_inline(lines, start):
    full = lines[start]
    depth = full.count('(') - full.count(')')
    j = start + 1
    while depth > 0 and j < len(lines):
        full += '\n' + lines[j]
        depth += lines[j].count('(') - lines[j].count(')')
        j += 1
    return full, j


def clean_import_aa(content):
    """Add import autoarray as aa if deflections code was added and aa not yet imported."""
    if 'aa.Array2D(' in content and 'import autoarray as aa' not in content:
        # Add after the autolens import
        content = re.sub(
            r'(import autolens as al\n)',
            r'\1import autoarray as aa\n',
            content,
            count=1
        )
    return content


def process_file(filepath):
    original = filepath.read_text(encoding='utf-8')

    # Only process if it has old API patterns
    old_markers = [
        'Visuals2D', 'MatPlot2D', 'figures_2d', 'visuals_2d', 'mat_plot_2d',
        'aplt.Figure', 'aplt.YTicks', 'aplt.XTicks', 'aplt.Title',
        'aplt.YLabel', 'aplt.XLabel', 'Plotter',
    ]
    if not any(m in original for m in old_markers):
        return False

    content = original
    content = fix_code_patterns(content)
    content = apply_docstring_subs(content)
    content = clean_import_aa(content)

    if content != original:
        filepath.write_text(content, encoding='utf-8')
        return True
    return False


SKIP_FILES = {
    'scripts/guides/plot/start_here.py',
    'scripts/guides/plot/examples/visuals.py',
    'scripts/guides/plot/examples/mat_plot.py',
    'scripts/guides/plot/examples/plotters.py',
    'scripts/guides/plot/examples/searches.py',
    'scripts/guides/plot/advanced/publication/image.py',
    'scripts/guides/plot/advanced/publication/subhalo_scan.py',
    'scripts/guides/plot/advanced/multi/MultiSubplots.py',
    'scripts/guides/plot/advanced/multi/MultiYX1DPlotter.py',
    'scripts/guides/plot/advanced/multi/MultiFigurePlotter.py',
    'scripts/guides/plot/advanced/plotters_double_einstein_ring.py',
    'scripts/guides/plot/advanced/plotters_pixelization.py',
    'scripts/howtolens/chapter_1_introduction/tutorial_0_visualization.py',
    'transform_plotting_api.py',
    'cleanup_remaining.py',
}


if __name__ == '__main__':
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/user/autolens_workspace')
    changed = 0
    for filepath in sorted(root.rglob('*.py')):
        rel = str(filepath.relative_to(root))
        if rel in SKIP_FILES:
            continue
        try:
            if process_file(filepath):
                changed += 1
                print(f"  {rel}")
        except Exception as e:
            print(f"  ERROR {rel}: {e}")
    print(f"\nChanged: {changed} files")
