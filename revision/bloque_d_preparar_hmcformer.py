"""
Bloque D de la revision PeerJ: crea una copia corregida de la replica de HMCFormer.

No modifica el original (Coat/Pappers/HMCFormer.py). Escribe
~/Cervical/revision_peerj/hmcformer/HMCFormer_train_only.py con cuatro cambios:
  1. La validacion cruzada usa solo las 541 imagenes de train (antes juntaba train y test).
  2. Una semilla (42) y 5 folds, como los demas baselines.
  3. Nombre y carpeta de salida nuevos, para no sobrescribir los resultados originales.
  4. Cada fold evalua su mejor checkpoint (el mismo que ya elegia el script) en las 225 de test
     y guarda las metricas en test_por_fold.csv y las predicciones por imagen.
Todo lo demas (arquitectura, epocas, optimizador, augmentation, seleccion del mejor epoch) queda igual.
"""
import os

ORIGEN = os.path.expanduser("~/Cervical/Coat/Pappers/HMCFormer.py")
DESTINO_DIR = os.path.expanduser("~/Cervical/revision_peerj/hmcformer")
DESTINO = os.path.join(DESTINO_DIR, "HMCFormer_train_only.py")

codigo = open(ORIGEN, encoding="utf-8").read()

CAMBIOS = [
    ('MODEL_NAME = "HMCFormer_replica_resized"',
     'MODEL_NAME = "HMCFormer_train_only"'),
    ('GRAPHICS_PATH = os.path.expanduser("~/Cervical/Coat/Pappers/graficas")',
     'GRAPHICS_PATH = os.path.expanduser("~/Cervical/revision_peerj/hmcformer/graficas")'),
    ("SEEDS = [42, 123, 999]",
     "SEEDS = [42]"),
    ('DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
     'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
     "TEST_X, TEST_Y = None, None  # revision PeerJ: test independiente"),
    ("    X_all = np.concatenate([X_train, X_test], axis=0)\n"
     "    y_all = np.concatenate([y_train, y_test], axis=0)\n",
     "    # revision PeerJ: la validacion cruzada usa solo el train; el test queda aparte\n"
     "    global TEST_X, TEST_Y\n"
     "    TEST_X, TEST_Y = X_test, y_test\n"
     "    X_all, y_all = X_train, y_train\n"),
    ("    # Guardar historial como CSV\n",
     "    # revision PeerJ: evaluar el mejor checkpoint del fold en el test independiente\n"
     "    test_ds = HMCFormerDataset(TEST_X, TEST_Y, transform=val_transform, apply_enhancement=True)\n"
     "    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False,\n"
     "                             num_workers=4, pin_memory=True, drop_last=False)\n"
     "    test_dir = os.path.join(fold_save_path, 'test')\n"
     "    ensure_dir(test_dir)\n"
     "    t_labels, t_preds, t_probs = full_evaluation(\n"
     "        model, test_loader, class_names=[f'Clase_{i}' for i in range(NUM_CLASSES)],\n"
     "        device=DEVICE, save_path=test_dir,\n"
     "        model_name=f'{MODEL_NAME}_fold{fold_id}_seed{seed}_TEST', save_csv=True)\n"
     "    t_m = compute_clinical_metrics(t_labels, t_preds, average='macro')\n"
     "    resumen = os.path.join(SAVE_GRAPHICS_PATH, 'test_por_fold.csv')\n"
     "    nuevo = not os.path.exists(resumen)\n"
     "    with open(resumen, 'a') as fh:\n"
     "        if nuevo:\n"
     "            fh.write('seed,fold,best_epoch,val_acc,test_acc,test_precision,test_recall,test_f1\\n')\n"
     "        fh.write(f\"{seed},{fold_id},{best_epoch},{best_val_acc:.2f},{t_m['accuracy']:.4f},\"\n"
     "                 f\"{t_m['precision']:.4f},{t_m['recall']:.4f},{t_m['f1']:.4f}\\n\")\n"
     "    pd.DataFrame({'y_true': np.asarray(t_labels), 'y_pred': np.asarray(t_preds)}).to_csv(\n"
     "        os.path.join(test_dir, 'predicciones_test.csv'), index_label='idx')\n"
     "    print(f\"[TEST] fold {fold_id} seed {seed}: acc {t_m['accuracy']:.4f} P {t_m['precision']:.4f} \"\n"
     "          f\"R {t_m['recall']:.4f} F1 {t_m['f1']:.4f}\", flush=True)\n"
     "\n"
     "    # Guardar historial como CSV\n"),
]

for viejo, nuevo in CAMBIOS:
    n = codigo.count(viejo)
    if n != 1:
        raise SystemExit(f"No se aplico ningun cambio: se esperaba 1 coincidencia y hay {n} de:\n{viejo}")
    codigo = codigo.replace(viejo, nuevo)

os.makedirs(DESTINO_DIR, exist_ok=True)
with open(DESTINO, "w", encoding="utf-8") as fh:
    fh.write(codigo)
print("Copia corregida creada en", DESTINO)
print("Cambios aplicados:", len(CAMBIOS))
