1����ȡ���ݣ� ʹ��pd.read_csv()���� train_df, test_df����
2���ϲ����ݣ�
label��ʹ��log1pƽ������train_df�е�label�õ�[y_train] -> �����Ҫ��expm1() �����
��ȡlabel�� ʹ��np.popȡ��label���ڵ���ģ���в���ʹ�ã��мǣ�ǧ��Ҫ��test_df��������
�ϲ�train&test�� ʹ��np.concat�ϲ�train_df, test_df�õ�[all_df]��һ��Ԥ����ʹ��
3�����������б�����
��ò���������Ϊ��ǩ�������ݣ��������壺 int -> string -> One-hot���루.get_dummies��->all_dfҲ����One-hot,�д�79�б�Ϊ303�У��õ�[all_dummy_df]
����ȱʧֵ��������ƽ��ֵ���棬���������.isnull().sum().sort_values(ascending=False) -> .mean() -> .fillna(meanֵ)
��׼�����ݣ������ʹ��, ����Է�one-hot�д���, all_df.dtypes != 'object' -> ��ֵ-ƽ��ֵ������ -> ok
4, ����ģ�ͣ�

Ridge Regression: ����Ridge & cross_val_score -> �ֱ��ȡѵ����dummy_train_df & ���Լ�dummy_test_df,
���ö���alpha����: ʹ��np.logspace -> ѭ�����ò�ͬ��alpha���õ����Է�������ͼ (neg_mean_squared_error) -> �õ�error��С�ĵط�
Random Forest: ����RandomForestRegressor -> ѭ����ͬmax_features, �õ����Է�������ͼ��neg_mean_squared_error�� -> �õ���С��error�ط�
5�����ģ�ͣ�

ʹ����õĲ�������������ģ�� ridge & ?RandomFroestRegressor
������ģ��ѵ������������.fit(X_train, y_train)
�����������X_test���õ�����Ԥ���� -> �������ƽ�� -> y_final
6���ύ���

data = pd.DataFrame({'id': test_df.index, 'SalePrice':y_final})
--------------------- 
���ߣ�sinat_15355869 
��Դ��CSDN 
ԭ�ģ�https://blog.csdn.net/sinat_15355869/article/details/79941945 
��Ȩ����������Ϊ����ԭ�����£�ת���븽�ϲ������ӣ�