import pandas as pd
from flask import Flask, request, render_template
from prediction_indonesian_food import recommend_recipes, print_recommendations

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = None
    if request.method == "POST":
        user_input = request.form["ingredients"]
        recommended_recipes = recommend_recipes(user_input)
        if recommended_recipes is not None:
            data = recommended_recipes[["Title", "Ingredients", "Steps", "URL"]]
            df = pd.DataFrame(data)
            recommendations = print_recommendations(df)
    return render_template("index.html", recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
