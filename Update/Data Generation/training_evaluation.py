# Function to train and evaluate
def evaluate_model(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"--- {name} Performance ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f} hours")
    print(f"R² Score: {r2:.4f}")
    
    # Threshold Check based on your proposal [cite: 45]
    if r2 > 0.70:
        print(" SUCCESS: R² meets the > 0.70 threshold.")
    else:
        print(" NOTE: Optimization needed to reach 0.70 threshold.")
    print("-" * 30)
    return model

# Train both
trained_lr = evaluate_model(lr_model, "Linear Regression")
trained_rf = evaluate_model(rf_model, "Random Forest")

# --- FEATURE IMPORTANCE EXTRACTION ---
# This helps explain to Daskan specifically *which* factors drive costs [cite: 47]
rf_feature_names = (numeric_features + 
                    list(trained_rf.named_steps['preprocessor']
                         .named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names_out(categorical_features)))

importances = trained_rf.named_steps['regressor'].feature_importances_

# Create a DataFrame for plotting
feature_imp_df = pd.DataFrame({'Feature': rf_feature_names, 'Importance': importances})
feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
plt.title('Feature Importance (Random Forest)')
plt.show()
