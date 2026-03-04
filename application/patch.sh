for f in src/Convolution*.c; do
  if grep -q "out_mult_in" "$f" && grep -q "out_shift_in" "$f" && grep -q "uint16_t out_shift = out_shift_in;" "$f" && ! grep -q "uint16_t out_mult" "$f"; then
    # Insert out_mult alias immediately before the out_shift alias
    perl -0777 -i -pe 's/\n(\s*)uint16_t out_shift = out_shift_in;/\n$1uint16_t out_mult = out_mult_in;\n$1uint16_t out_shift = out_shift_in;/g' "$f"
  fi
done
for f in src/Convolution*.c; do
  if grep -q "out_mult_in" "$f" && grep -q "uint16_t out_shift = out_shift_in;" "$f"; then
    grep -q "uint16_t out_mult" "$f" || echo "STILL MISSING out_mult in $f"
  fi
done