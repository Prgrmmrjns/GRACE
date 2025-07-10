import cairosvg
import argparse
import os

def convert_svg_to_png(svg_path):
    """
    Converts an SVG file to a PNG file.

    The output PNG file will have the same name and be in the same directory
    as the input SVG file.
    """
    if not svg_path.lower().endswith('.svg'):
        print("Error: Input file must be an SVG file.")
        return

    if not os.path.exists(svg_path):
        print(f"Error: File not found at '{svg_path}'")
        return

    # Define output path
    png_path = os.path.splitext(svg_path)[0] + '.png'

    try:
        print(f"Converting '{svg_path}' to '{png_path}'...")
        cairosvg.svg2png(url=svg_path, write_to=png_path)
        print("Conversion successful!")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert an SVG file to a PNG file.')
    parser.add_argument('svg_file', type=str, help='The path to the input SVG file.')
    
    args = parser.parse_args()
    
    convert_svg_to_png(args.svg_file) 