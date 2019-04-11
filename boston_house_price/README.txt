1，读取数据： 使用pd.read_csv()导入 train_df, test_df数据
2，合并数据：
label：使用log1p平滑处理train_df中的label得到[y_train] -> 最后需要用expm1() 变回来
提取label： 使用np.pop取出label后期导入模型中测试使用，切记：千万不要动test_df测试数据
合并train&test： 使用np.concat合并train_df, test_df得到[all_df]，一会预处理使用
3，处理数据中变量：
最好不用数字作为标签划分数据，出现歧义： int -> string -> One-hot编码（.get_dummies）->all_df也采用One-hot,列从79列变为303列，得到[all_dummy_df]
处理缺失值，这里用平均值代替，看情况处理：.isnull().sum().sort_values(ascending=False) -> .mean() -> .fillna(mean值)
标准化数据，看情况使用, 这里对非one-hot列处理, all_df.dtypes != 'object' -> 数值-平均值／方差 -> ok
4, 建立模型：

Ridge Regression: 导入Ridge & cross_val_score -> 分别获取训练集dummy_train_df & 测试集dummy_test_df,
采用多组alpha测试: 使用np.logspace -> 循环调用不同的alpha，得到测试分数，画图 (neg_mean_squared_error) -> 得到error最小的地方
Random Forest: 导入RandomForestRegressor -> 循环不同max_features, 得到测试分数，画图（neg_mean_squared_error） -> 得到最小的error地方
5，组合模型：

使用最好的参数，建立两种模型 ridge & ?RandomFroestRegressor
用两种模型训练，输入数据.fit(X_train, y_train)
输入测试数据X_test，得到两组预测结果 -> 这里简单求平均 -> y_final
6，提交结果

data = pd.DataFrame({'id': test_df.index, 'SalePrice':y_final})
--------------------- 
作者：sinat_15355869 
来源：CSDN 
原文：https://blog.csdn.net/sinat_15355869/article/details/79941945 
版权声明：本文为博主原创文章，转载请附上博文链接！