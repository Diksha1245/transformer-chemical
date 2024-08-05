# Define hyperparameters
num_layers = 24
d_model = 512
dff = 512
num_heads = 8
input_vocab_size = num_features  # Number of features in X
target_vocab_size = 4  # Number of target variables
dropout_rate = 0.2

# Create a transformer model instance
transformer = Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input=1000, pe_target=1000, rate=dropout_rate)

# Compile the model
transformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),loss='mse', metrics=['mae', 'accuracy'])


# Train the model
history = transformer.fit(X_train_input, y_train_input, epochs=20,batch_size=128, verbose=1)

# Inverse transform predictions to get actual values
train_predictions_scaled = transformer.predict(X_train_input)
test_predictions_scaled = transformer.predict(X_test_input)

# Reshape predictions to match the shape of scaled targets
train_predictions_scaled = train_predictions_scaled.reshape(-1, y_train_scaled.shape[1])
test_predictions_scaled = test_predictions_scaled.reshape(-1, y_test_scaled.shape[1])

# Inverse transform to get actual values
train_predictions = scaler_y.inverse_transform(train_predictions_scaled)
test_predictions = scaler_y.inverse_transform(test_predictions_scaled)

# Ensure correct reshaping for actual values
actual_train_values = scaler_y.inverse_transform(y_train_input[:, -1, :])
actual_test_values = scaler_y.inverse_transform(y_test_input[:, -1, :])

# Reshape predictions to match the shape of actual values
train_predictions = train_predictions[:actual_train_values.shape[0], :]
test_predictions = test_predictions[:actual_test_values.shape[0], :]

# Print shapes to verify alignment
print(f"Shapes after alignment - train_predictions: {train_predictions.shape}, actual_train_values: {actual_train_values.shape}")
print(f"Shapes after alignment - test_predictions: {test_predictions.shape}, actual_test_values: {actual_test_values.shape}")

# Print evaluation metrics 
for i, col in enumerate(['Level1', 'Level2', 'PCPV', 'DT']): 
  mae_train_i = np.mean(np.abs(train_predictions[:, i] - actual_train_values[:, i])) 
  mae_test_i = np.mean(np.abs(test_predictions[:, i] - actual_test_values[:, i])) 
  print(f"\nMean Absolute Error (MAE) for {col} (Training): {mae_train_i:.2f}") 
  print(f"Mean Absolute Error (MAE) for {col} (Testing): {mae_test_i:.2f}") 
# Overall Mean Absolute Error (MAE) 
mae_train = np.mean(np.abs(train_predictions - actual_train_values), axis=0) 
mae_test = np.mean(np.abs(test_predictions - actual_test_values), axis=0) 
print(f"\nOverall Mean Absolute Error (MAE) for Training Data: {mae_train}") 
print(f"\nOverall Mean Absolute Error (MAE) for Testing Data: {mae_test}") 



# Print actual vs predicted values for training data
train_results = pd.DataFrame({
    'Actual_Level1': actual_train_values[:, 0],
    'Predicted_Level1': train_predictions[:, 0],
    'Actual_Level2': actual_train_values[:, 1],
    'Predicted_Level2': train_predictions[:, 1],
    'Actual_PCPV': actual_train_values[:, 2],
    'Predicted_PCPV': train_predictions[:, 2],
    'Actual_DT': actual_train_values[:, 3],
    'Predicted_DT': train_predictions[:, 3]
})

print("Training Data Results:")
print(train_results)

# Print actual vs predicted values for testing data
test_results = pd.DataFrame({
    'Actual_Level1': actual_test_values[:, 0],
    'Predicted_Level1': test_predictions[:, 0],
    'Actual_Level2': actual_test_values[:, 1],
    'Predicted_Level2': test_predictions[:, 1],
    'Actual_PCPV': actual_test_values[:, 2],
    'Predicted_PCPV': test_predictions[:, 2],
    'Actual_DT': actual_test_values[:, 3],
    'Predicted_DT': test_predictions[:, 3]
})

print("Testing Data Results:")
print(test_results)

# Print evaluation metrics
for i, col in enumerate(['Level1', 'Level2', 'PCPV', 'DT']):
    mae_train_i = np.mean(np.abs(train_predictions[:, i] - actual_train_values[:, i]))
    mae_test_i = np.mean(np.abs(test_predictions[:, i] - actual_test_values[:, i]))
    print(f"\nMean Absolute Error (MAE) for {col} (Training): {mae_train_i:.2f}")
    print(f"Mean Absolute Error (MAE) for {col} (Testing): {mae_test_i:.2f}")

# Overall Mean Absolute Error (MAE)
mae_train = np.mean(np.abs(train_predictions - actual_train_values), axis=0)
mae_test = np.mean(np.abs(test_predictions - actual_test_values), axis=0)
print(f"\nOverall Mean Absolute Error (MAE) for Training Data: {mae_train}")
print(f"Overall Mean Absolute Error (MAE) for Testing Data: {mae_test}")
test_results.to_csv("test_results4.csv",inplace=False)
train_results.to_csv("train_results4.csv",inplace=False)
