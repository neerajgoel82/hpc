#!/bin/bash

echo "=========================================="
echo "CUDA Learning Curriculum Verification"
echo "=========================================="
echo ""

# Count notebooks by phase
echo "Notebooks by Phase:"
echo "-------------------"
for i in {1..9}; do
    count=$(find notebooks/phase$i -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')
    echo "Phase $i: $count notebooks"
done

echo ""
echo "Total Notebooks:"
echo "----------------"
total=$(find notebooks -name "*.ipynb" | wc -l | tr -d ' ')
echo "Total: $total notebooks"

echo ""
echo "Expected: 55 curriculum notebooks + 1 setup = 56 total"
echo ""

if [ "$total" -eq 56 ]; then
    echo "✅ SUCCESS: All notebooks created!"
else
    echo "⚠️  WARNING: Expected 56 notebooks, found $total"
fi

echo ""
echo "Directory Structure:"
echo "--------------------"
tree -L 2 notebooks/ 2>/dev/null || find notebooks/ -type d | sort

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
