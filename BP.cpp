// BP.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#include<time.h>

#define COUNT 6
#define IN_NUM 3		//输入层个数
#define OUT_NUM 2		//输出层个数
#define numHidden 30	//隐藏层个数

double weight_hidden[IN_NUM][numHidden];		//输入层到隐藏层的权重
double bias_hidden[numHidden];					//偏置量
double weight_output[numHidden][OUT_NUM];		//隐藏层到输出层的权重
double bias_output[OUT_NUM];					
double learnRate = 0.8;							//学习数率
double accuracy = 0.000001;						//精度控制
int maxLoopCount = 1000000;						//最大循环次数

double fnet(double net)							//sigmod函数
{
	return 1 / (1 + exp(-net));
}
double dfnet(double net)						//sigmod函数求导
{
	return net * (1 - net);
}

//BP神经网络初始化
int InitBp()
{
	int i, j;
	srand((unsigned)time(NULL));
	for (i = 0; i < IN_NUM; i++)
		for (j = 0; j < numHidden; j++)
		{
			weight_hidden[i][j] = rand() / (double)(RAND_MAX)-0.5;
			bias_hidden[j] = rand() / (double)(RAND_MAX)-0.5;
		}
	for (i = 0; i < numHidden; i++)
		for (j = 0; j < OUT_NUM; j++)
		{
			weight_output[i][j] = rand() / (double)(RAND_MAX)-0.5;
			bias_output[j] = rand() / (double)(RAND_MAX)-0.5;
		}
	return 1;
}

//训练BP网络
int TrainBp(float x[COUNT][IN_NUM], float y[COUNT][OUT_NUM])
{
	double delta_hidden[numHidden], delta_output[OUT_NUM];
	double output_hidden[numHidden], output_output[OUT_NUM];
	int i, j, k, n;
	double temp;

	double loss = accuracy + 1;   //目的是为了让loss大于f，能进行下面的循环  
	for (n = 0; loss > accuracy && n < maxLoopCount; n++)
	{
		loss = 0;
		for (i = 0; i < COUNT; i++)
		{
			//前向计算  
			for (k = 0; k < numHidden; k++)
			{
				temp = 0;
				for (j = 0; j < IN_NUM; j++)
					temp += x[i][j] * weight_hidden[j][k];
				output_hidden[k] = fnet(temp + bias_hidden[k]);
			}
			for (k = 0; k < OUT_NUM; k++)
			{
				temp = 0;
				for (j = 0; j < numHidden; j++)
					temp += output_hidden[j] * weight_output[j][k];
				output_output[k] = fnet(temp + bias_output[k]);
			}

			//计算误差  
			for (j = 0; j < OUT_NUM; j++)
				loss += 0.5 * (y[i][j] - output_output[j]) * (y[i][j] - output_output[j]);

			//反向计算  
			for (j = 0; j < OUT_NUM; j++)
				delta_output[j] = (y[i][j] - output_output[j]) * dfnet(output_output[j]);

			for (j = 0; j < numHidden; j++)
				for (k = 0; k < OUT_NUM; k++)
					weight_output[j][k] += learnRate * delta_output[k] * output_hidden[j];
			for (k = 0; k < OUT_NUM; k++)
				bias_output[k] += learnRate * delta_output[k];


			for (j = 0; j < numHidden; j++)
			{
				temp = 0;
				for (k = 0; k < OUT_NUM; k++)
					temp += weight_output[j][k] * delta_output[k];
				delta_hidden[j] = temp * dfnet(output_hidden[j]);
			}

			for (j = 0; j < IN_NUM; j++)
				for (k = 0; k < numHidden; k++)
					weight_hidden[j][k] += learnRate * delta_hidden[k] * x[i][j]; //隐藏层权矩阵  
			for (k = 0; k < numHidden; k++)
				bias_hidden[k] += learnRate * delta_hidden[k];
		}
		if (n % 10 == 0)
			printf("误差 : %f\n", loss);          //每训练10次输出一次误差结果  
	}
	printf("总共循环次数：%d\n", n);

	printf("bp网络训练结束！\n");

	return 1;
}

int UseBp()
{
	float Input[IN_NUM];
	double output_hidden[numHidden];			//隐藏层输出
	double output_output[OUT_NUM];				//输出层输出
	while (1)		//持续执行，除非中断程序
	{
		printf("请输入3个数： \n");
		int i, j;
		for (i = 0; i < IN_NUM; i++)
			scanf("%f", &Input[i]);

		double temp;
		for (i = 0; i < numHidden; i++)
		{
			temp = 0;
			for (j = 0; j < IN_NUM; j++)
				temp += Input[j] * weight_hidden[j][i];
			output_hidden[i] = fnet(temp + bias_hidden[i]);
		}
		for (i = 0; i < OUT_NUM; i++)
		{
			temp = 0;
			for (j = 0; j < numHidden; j++)
				temp += output_hidden[j] * weight_hidden[j][i];
			output_output[i] = fnet(temp + bias_output[i]);
		}

		printf("结果：  ");
		for (i = 0; i < OUT_NUM; i++)
			printf("%.3f", output_output[i]);
		printf("\n");
	}
	return 1;
}
int _tmain(int argc, _TCHAR* argv[])
{
	//x---输入向量，y---输出向量  
	//输入：0.8, 0.5,   0---------输出0,1  
	//输入：0.9, 0.7, 0.3---------输出0,1  
	//输入：  1, 0.8, 0.5---------输出0,1  
	//输入：  0, 0.2, 0.3---------输出1,0                      
	//输入：0.2, 0.1, 1.3---------输出1,0                              
	//输入：0.2, 0.7, 0.8---------输出1,0      
	float x[COUNT][IN_NUM] = { { 0.8, 0.5, 0 },
	{ 0.9, 0.7, 0.3 },
	{ 1, 0.8, 0.5 },
	{ 0, 0.2, 0.3 },
	{ 0.2, 0.1, 1.3 },
	{ 0.2, 0.7, 0.8 } }; //训练样本  
	float y[COUNT][OUT_NUM] = { { 0, 1 },
	{ 0, 1 },
	{ 0, 1 },
	{ 1, 0 },
	{ 1, 0 },
	{ 1, 0 } };      //理想输出  

	InitBp();                    //初始化bp网络结构  
	TrainBp(x, y);             //训练bp神经网络  
	UseBp();                     //测试bp神经网络  
	return 1;
}

