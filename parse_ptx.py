#!/usr/bin/env python3
"""
PTX File Parser - Replace file numbers in .loc directives with actual file paths
and optionally include source code lines
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_source_file(file_path: str) -> Optional[List[str]]:
    """
    Load a source file and return its lines.
    Returns None if file cannot be read.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readlines()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return None


def parse_ptx(input_file, output_file=None):
    """
    Parse PTX file and replace file numbers with actual file paths in .loc directives.
    
    Args:
        input_file: Path to input PTX file
        output_file: Path to output file (optional, defaults to input_file + '.parsed')
    """
    # Dictionary to store file number -> file path mapping
    file_map = {}
    
    # Read the PTX file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # First pass: collect all .file directives
    file_pattern = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"')
    
    for line in lines:
        match = file_pattern.match(line)
        if match:
            file_num = int(match.group(1))
            file_path = match.group(2)
            file_map[file_num] = file_path
            print(f"Found file {file_num}: {file_path}")
    
    print(f"\nTotal files found: {len(file_map)}\n")
    
    # Second pass: replace file numbers in .loc directives
    # .loc pattern: .loc <file_num> <line> <col> [, additional info]
    loc_pattern = re.compile(r'^(\s*\.loc\s+)(\d+)(\s+\d+\s+\d+.*)')
    
    output_lines = []
    replacements = 0
    
    for line in lines:
        match = loc_pattern.match(line)
        if match:
            prefix = match.group(1)
            file_num = int(match.group(2))
            suffix = match.group(3)
            
            if file_num in file_map:
                # Replace file number with file path (shortened for readability)
                file_path = file_map[file_num]
                # Optionally shorten the path for better readability
                short_path = Path(file_path).name
                new_line = f'{prefix}"{short_path}"{suffix}\n'
                output_lines.append(new_line)
                replacements += 1
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    
    print(f"Replaced {replacements} .loc directives\n")
    
    # Write output
    if output_file is None:
        output_file = str(input_file) + '.parsed'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"Output written to: {output_file}")
    
    return file_map, replacements


def parse_ptx_full_path(input_file, output_file=None):
    """
    Parse PTX file and replace file numbers with FULL file paths in .loc directives.
    
    Args:
        input_file: Path to input PTX file
        output_file: Path to output file (optional, defaults to input_file + '.parsed')
    """
    # Dictionary to store file number -> file path mapping
    file_map = {}
    
    # Read the PTX file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # First pass: collect all .file directives
    file_pattern = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"')
    
    for line in lines:
        match = file_pattern.match(line)
        if match:
            file_num = int(match.group(1))
            file_path = match.group(2)
            file_map[file_num] = file_path
    
    print(f"Total files found: {len(file_map)}\n")
    
    # Second pass: replace file numbers in .loc directives
    loc_pattern = re.compile(r'^(\s*\.loc\s+)(\d+)(\s+\d+\s+\d+.*)')
    
    output_lines = []
    replacements = 0
    
    for line in lines:
        match = loc_pattern.match(line)
        if match:
            prefix = match.group(1)
            file_num = int(match.group(2))
            suffix = match.group(3)
            
            if file_num in file_map:
                # Replace file number with FULL file path
                file_path = file_map[file_num]
                new_line = f'{prefix}"{file_path}"{suffix}\n'
                output_lines.append(new_line)
                replacements += 1
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    
    print(f"Replaced {replacements} .loc directives\n")
    
    # Write output
    if output_file is None:
        output_file = str(input_file) + '.full.parsed'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"Output written to: {output_file}")
    
    return file_map, replacements


def parse_ptx_with_source(input_file, output_file=None, max_line_length=100):
    """
    Parse PTX file and add source code comments next to .loc directives.
    
    Args:
        input_file: Path to input PTX file
        output_file: Path to output file (optional, defaults to input_file + '.annotated')
        max_line_length: Maximum length of source code line to include
    """
    # Dictionary to store file number -> file path mapping
    file_map = {}
    # Dictionary to store file path -> source lines mapping
    source_cache = {}
    
    # Read the PTX file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # First pass: collect all .file directives
    file_pattern = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"')
    
    for line in lines:
        match = file_pattern.match(line)
        if match:
            file_num = int(match.group(1))
            file_path = match.group(2)
            file_map[file_num] = file_path
            
            # Load source file
            if Path(file_path).exists():
                source_lines = load_source_file(file_path)
                if source_lines:
                    source_cache[file_num] = source_lines
                    print(f"Loaded file {file_num}: {file_path} ({len(source_lines)} lines)")
                else:
                    print(f"Could not load file {file_num}: {file_path}")
            else:
                print(f"File not found {file_num}: {file_path}")
    
    print(f"\nTotal files found: {len(file_map)}")
    print(f"Source files loaded: {len(source_cache)}\n")
    
    # Second pass: process .loc directives and add source code comments
    # .loc pattern: .loc <file_num> <line> <col> [, additional info]
    loc_pattern = re.compile(r'^(\s*)\.loc\s+(\d+)\s+(\d+)\s+(\d+)(.*)')
    
    output_lines = []
    replacements = 0
    
    for line in lines:
        match = loc_pattern.match(line)
        if match:
            indent = match.group(1)
            file_num = int(match.group(2))
            line_num = int(match.group(3))
            col_num = int(match.group(4))
            rest = match.group(5)
            
            if file_num in file_map:
                # Get filename
                file_path = file_map[file_num]
                short_name = Path(file_path).name
                
                # Create new .loc line with filename
                new_loc = f"//[]({file_path}:{line_num})\n"
                output_lines.append(new_loc)
                
                # Add source code comment if available
                if file_num in source_cache:
                    source_lines = source_cache[file_num]
                    if 0 < line_num <= len(source_lines):
                        source_line = source_lines[line_num - 1].rstrip()
                        # Trim leading whitespace but preserve some indentation info
                        source_line_stripped = source_line.lstrip()
                        
                        # Truncate if too long
                        if len(source_line_stripped) > max_line_length:
                            source_line_stripped = source_line_stripped[:max_line_length] + "..."
                        
                        if source_line_stripped:
                            comment = f'//{indent}{source_line_stripped}\n'
                            output_lines.append(comment)
                
                replacements += 1
            else:
                output_lines.append(line)
        else:
            output_lines.append(line)
    
    print(f"Annotated {replacements} .loc directives\n")
    
    # Write output
    if output_file is None:
        output_file = str(input_file) + '.annotated.ptx'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)
    
    print(f"Output written to: {output_file}")
    
    return file_map, replacements


def show_file_stats(input_file):
    """
    Show statistics about files referenced in PTX file.
    """
    file_map = {}
    loc_count = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Collect .file directives
    file_pattern = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"')
    for line in lines:
        match = file_pattern.match(line)
        if match:
            file_num = int(match.group(1))
            file_path = match.group(2)
            file_map[file_num] = file_path
            loc_count[file_num] = 0
    
    # Count .loc directives per file
    loc_pattern = re.compile(r'^\s*\.loc\s+(\d+)\s+')
    for line in lines:
        match = loc_pattern.match(line)
        if match:
            file_num = int(match.group(1))
            if file_num in loc_count:
                loc_count[file_num] += 1
    
    # Print statistics
    print("=" * 80)
    print("PTX File Statistics")
    print("=" * 80)
    print(f"Total files: {len(file_map)}")
    print(f"\nFile references (.loc count):")
    print("-" * 80)
    
    for file_num in sorted(file_map.keys()):
        count = loc_count[file_num]
        file_path = file_map[file_num]
        short_name = Path(file_path).name
        print(f"  [{file_num:2d}] {count:5d} refs - {short_name}")
        print(f"       {file_path}")
    
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python parse_ptx.py <ptx_file>                    # Parse with short filenames")
        print("  python parse_ptx.py <ptx_file> --source           # Parse and include source code")
        print("  python parse_ptx.py <ptx_file> --source <output>  # Specify output file")
        print("  python parse_ptx.py <ptx_file> --full             # Parse with full paths")
        print("  python parse_ptx.py <ptx_file> --stats            # Show statistics only")
        print("  python parse_ptx.py <ptx_file> <output>           # Specify output file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    if len(sys.argv) > 2 and sys.argv[2] == '--stats':
        show_file_stats(input_file)
    elif len(sys.argv) > 2 and sys.argv[2] == '--source':
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        parse_ptx_with_source(input_file, output_file)
    elif len(sys.argv) > 2 and sys.argv[2] == '--full':
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        parse_ptx_full_path(input_file, output_file)
    else:
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        parse_ptx_with_source(input_file, output_file)

