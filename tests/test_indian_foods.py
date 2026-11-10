# -*- coding: utf-8 -*-
import requests
import json

test_dishes = [
    ("Chicken Dum Biryani", {"Chicken Dum Biryani": 200}),
    ("Steamed Rice with Sambar", {"Steamed Rice": 150, "Sambar": 120}),
    ("Roti and Dal", {"Phulka": 50, "Dal Tadka": 120}),
    ("South Indian Breakfast", {"Idli": 100, "Sambar": 100, "Coconut Chutney": 30}),
    ("Paneer Meal", {"Palak Paneer": 150, "Roti": 60, "Curd": 100}),
]

print("========================================================")
print("  TESTING INDIAN FOOD PREDICTIONS (RETRAINED CATBOOST)  ")
print("========================================================\n")

for dish_title, ingredients in test_dishes:
    payload = {"dish": dish_title, "ingredients": ingredients}
    r = requests.post("http://127.0.0.1:8000/predict/vision", json=payload, timeout=10)
    print(f"Dish: {dish_title}")
    if r.status_code == 200:
        data = r.json()
        totals = data.get("recipe_totals", {})
        print(f"  Total Weight: {totals.get('total_weight_g')}g | Energy: {totals.get('energy_kcal')} kcal | Carbs: {totals.get('carb_g')}g | Protein: {totals.get('protein_g')}g | Fat: {totals.get('fat_g')}g | Fibre: {totals.get('fibre_g')}g")
        for item in data.get("nutrition", []):
            print(f"   * {item['ingredient']:22s} ({item['weight_g']}g): {item['energy_kcal']} kcal, {item['protein_g']}g prot, {item['carb_g']}g carb, {item['fat_g']}g fat, {item['fibre_g']}g fib")
    else:
        print(f"  Failed: {r.status_code} {r.text}")
    print()

