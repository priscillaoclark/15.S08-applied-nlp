def create_lasso_regression_model():
    K.clear_session()
    lasso_regularizer = l1_l2(l1=0.1, l2=0)
    model = Sequential()
    inputs = Input(shape=(X_train.shape[1],))
    model.add(inputs)
    model.add(Dense(1, activation='linear', kernel_regularizer=lasso_regularizer))
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

####

from keras.callbacks import ModelCheckpoint, EarlyStopping

# Use callbacks to stop training early if the loss on validation data does not decrease
keras_lin_reg_lasso_best = create_lasso_regression_model()
checkpoint_cb_lasso = ModelCheckpoint("keras_lasso.keras", save_best_only=True)
early_stopping_cb_lasso = EarlyStopping(patience=10, restore_best_weights=True)
history_lasso = keras_lin_reg_lasso_best.fit(X_train, y_train, epochs=2000, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb_lasso, early_stopping_cb_lasso])

# Plot the training history
history_df_lasso = pd.DataFrame(history_lasso.history)
history_df_lasso[['loss', 'val_loss']].plot()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.grid(True)
plt.show()

####

# Predict on the test set
y_pred_keras_lasso = keras_lin_reg_lasso_best.predict(X_test)

# Calculate the Mean Squared Error
mse_keras_lasso = mean_squared_error(y_test, y_pred_keras_lasso)
print(f"Mean Squared Error (Keras - Ridge): {mse_keras_lasso}")

# Calculate the R2 score
r2_keras_lasso = r2_score(y_test, y_pred_keras_lasso)
print(f"R2 Score (Keras): {r2_keras_lasso}")

#####

# Get the coefficients from the Keras model
keras_lasso_coefs = keras_lin_reg_lasso_best.layers[0].get_weights()[0].flatten()

# Compare the coefficients
comparison_df_lasso_keras = pd.DataFrame({
    'Real Coefficients': coefs,
    'Keras LASSO Coefficients': keras_lasso_coefs
})

print(comparison_df_lasso_keras)