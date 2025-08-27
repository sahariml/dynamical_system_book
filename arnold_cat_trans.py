import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def arnold_map(x, y, width, height):
    """Apply Arnold's cat map transformation"""
    nx = (2 * x + y) % width
    ny = (x + y) % height
    return nx, ny

def process_image(image, iterations=1):
    """Apply Arnold's cat map transformation to an image"""
    width, height = image.size
    img_array = np.array(image)
    
    for _ in range(iterations):
        new_array = np.zeros_like(img_array)
        for x in range(width):
            for y in range(height):
                nx, ny = arnold_map(x, y, width, height)
                # Handle y-coordinate flip for image coordinate system
                new_y = height - int(ny) - 1
                new_array[new_y, int(nx)] = img_array[y, x]
        img_array = new_array
    
    return Image.fromarray(img_array)

def main():
    # Configuration
    input_path = "input.png"
    output_prefix = "arnold_cat"
    iterations = 4
    
    # Verify input exists
    if not os.path.exists(input_path):
        print(f"Error: Input image '{input_path}' not found in current directory")
        return
    
    try:
        # Load and convert to RGB (supports EPS format)
        original = Image.open(input_path).convert('RGB')
        print(f"Processing image: {input_path} ({original.size[0]}x{original.size[1]})")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Process and save each iteration
    results = []
    current = original
    
    for i in range(1, iterations + 1):
        print(f"Applying iteration {i}...")
        current = process_image(current, 1)
        
        # Save as EPS with proper format
        output_path = f"{output_prefix}_iteration_{i}.eps"
        current.save(output_path, format="EPS")
        print(f"Saved: {output_path}")
        results.append(current)
    
    # Save final result
    final_output = f"{output_prefix}_final.eps"
    results[-1].save(final_output, format="EPS")
    print(f"\nFinal result saved as: {final_output}")
    
    # Generate comparison figure
    plt.figure(figsize=(12, 4))
    plt.subplot(1, iterations + 1, 1)
    plt.imshow(np.array(original))
    plt.title("Original")
    plt.axis('off')
    
    for i, img in enumerate(results):
        plt.subplot(1, iterations + 1, i + 2)
        plt.imshow(np.array(img))
        plt.title(f"Iteration {i+1}")
        plt.axis('off')
        #plt.show()
    plt.tight_layout()
    plt.savefig("arnold_cat_comparison.png", dpi=150, bbox_inches='tight')
    print("Comparison figure saved as: arnold_cat_comparison.png")

if __name__ == "__main__":
    main()