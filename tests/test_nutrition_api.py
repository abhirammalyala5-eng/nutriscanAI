import requests, json, time
time.sleep(4)

# Test 1: Biryani (dish context)
r1 = requests.post('http://localhost:8000/predict/vision', json={
    'dish': 'Biryani',
    'ingredients': {'Rice': 440, 'Chicken': 90, 'Onion': 30}
}, timeout=30).json()

# Test 2: No dish — smart ingredient defaults kick in
r2 = requests.post('http://localhost:8000/predict/vision', json={
    'ingredients': {'Rice': 200, 'Chicken': 150, 'Apple': 100, 'Salmon': 80}
}, timeout=30).json()

print("=== Test 1: Biryani (dish context) ===")
for i in r1['nutrition']:
    print(f"  {i['ingredient']:15} -> {i['energy_kcal']} kcal")

print()
print("=== Test 2: No dish (smart defaults) ===")
for i in r2['nutrition']:
    print(f"  {i['ingredient']:15} -> {i['energy_kcal']} kcal")
