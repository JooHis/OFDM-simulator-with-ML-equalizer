½
éº
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ïÂ
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

module_wrapper/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*,
shared_namemodule_wrapper/dense/kernel

/module_wrapper/dense/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper/dense/kernel*
_output_shapes

:@*
dtype0

module_wrapper/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namemodule_wrapper/dense/bias

-module_wrapper/dense/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper/dense/bias*
_output_shapes
:@*
dtype0

module_wrapper_1/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!module_wrapper_1/dense_1/kernel

3module_wrapper_1/dense_1/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper_1/dense_1/kernel*
_output_shapes

:@*
dtype0

module_wrapper_1/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namemodule_wrapper_1/dense_1/bias

1module_wrapper_1/dense_1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_1/dense_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
 
"Adam/module_wrapper/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"Adam/module_wrapper/dense/kernel/m

6Adam/module_wrapper/dense/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/module_wrapper/dense/kernel/m*
_output_shapes

:@*
dtype0

 Adam/module_wrapper/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/module_wrapper/dense/bias/m

4Adam/module_wrapper/dense/bias/m/Read/ReadVariableOpReadVariableOp Adam/module_wrapper/dense/bias/m*
_output_shapes
:@*
dtype0
¨
&Adam/module_wrapper_1/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/module_wrapper_1/dense_1/kernel/m
¡
:Adam/module_wrapper_1/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_1/dense_1/kernel/m*
_output_shapes

:@*
dtype0
 
$Adam/module_wrapper_1/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/module_wrapper_1/dense_1/bias/m

8Adam/module_wrapper_1/dense_1/bias/m/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_1/dense_1/bias/m*
_output_shapes
:*
dtype0
 
"Adam/module_wrapper/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"Adam/module_wrapper/dense/kernel/v

6Adam/module_wrapper/dense/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/module_wrapper/dense/kernel/v*
_output_shapes

:@*
dtype0

 Adam/module_wrapper/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" Adam/module_wrapper/dense/bias/v

4Adam/module_wrapper/dense/bias/v/Read/ReadVariableOpReadVariableOp Adam/module_wrapper/dense/bias/v*
_output_shapes
:@*
dtype0
¨
&Adam/module_wrapper_1/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*7
shared_name(&Adam/module_wrapper_1/dense_1/kernel/v
¡
:Adam/module_wrapper_1/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_1/dense_1/kernel/v*
_output_shapes

:@*
dtype0
 
$Adam/module_wrapper_1/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/module_wrapper_1/dense_1/bias/v

8Adam/module_wrapper_1/dense_1/bias/v/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_1/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
 .
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Û-
valueÑ-BÎ- BÇ-
§
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

_module
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

_module
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*

!iter

"beta_1

#beta_2
	$decay
%learning_rate&m`'ma(mb)mc&vd've(vf)vg*
 
&0
'1
(2
)3*
 
&0
'1
(2
)3*
* 
°
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

/serving_default* 
* 
* 
* 

0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
¦

&kernel
'bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
*9&call_and_return_all_conditional_losses
:__call__*

&0
'1*

&0
'1*
* 

;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
¦

(kernel
)bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
*D&call_and_return_all_conditional_losses
E__call__*

(0
)1*

(0
)1*
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmodule_wrapper/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmodule_wrapper/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmodule_wrapper_1/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmodule_wrapper_1/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

K0
L1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

&0
'1*

&0
'1*

5regularization_losses
Mmetrics
Nnon_trainable_variables

Olayers
6	variables
7trainable_variables
Player_regularization_losses
Qlayer_metrics
:__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 

(0
)1*

(0
)1*

@regularization_losses
Rmetrics
Snon_trainable_variables

Tlayers
A	variables
Btrainable_variables
Ulayer_regularization_losses
Vlayer_metrics
E__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
8
	Wtotal
	Xcount
Y	variables
Z	keras_api*
H
	[total
	\count
]
_fn_kwargs
^	variables
_	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

W0
X1*

Y	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

[0
\1*

^	variables*
~x
VARIABLE_VALUE"Adam/module_wrapper/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/module_wrapper/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/module_wrapper_1/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/module_wrapper_1/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/module_wrapper/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/module_wrapper/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/module_wrapper_1/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/module_wrapper_1/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_flatten_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
¸
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_inputmodule_wrapper/dense/kernelmodule_wrapper/dense/biasmodule_wrapper_1/dense_1/kernelmodule_wrapper_1/dense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_216045
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ù	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/module_wrapper/dense/kernel/Read/ReadVariableOp-module_wrapper/dense/bias/Read/ReadVariableOp3module_wrapper_1/dense_1/kernel/Read/ReadVariableOp1module_wrapper_1/dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp6Adam/module_wrapper/dense/kernel/m/Read/ReadVariableOp4Adam/module_wrapper/dense/bias/m/Read/ReadVariableOp:Adam/module_wrapper_1/dense_1/kernel/m/Read/ReadVariableOp8Adam/module_wrapper_1/dense_1/bias/m/Read/ReadVariableOp6Adam/module_wrapper/dense/kernel/v/Read/ReadVariableOp4Adam/module_wrapper/dense/bias/v/Read/ReadVariableOp:Adam/module_wrapper_1/dense_1/kernel/v/Read/ReadVariableOp8Adam/module_wrapper_1/dense_1/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_216223
°
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratemodule_wrapper/dense/kernelmodule_wrapper/dense/biasmodule_wrapper_1/dense_1/kernelmodule_wrapper_1/dense_1/biastotalcounttotal_1count_1"Adam/module_wrapper/dense/kernel/m Adam/module_wrapper/dense/bias/m&Adam/module_wrapper_1/dense_1/kernel/m$Adam/module_wrapper_1/dense_1/bias/m"Adam/module_wrapper/dense/kernel/v Adam/module_wrapper/dense/bias/v&Adam/module_wrapper_1/dense_1/kernel/v$Adam/module_wrapper_1/dense_1/bias/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_216296øÓ
Ô
Á
F__inference_sequential_layer_call_and_return_conditional_losses_216030

inputsE
3module_wrapper_dense_matmul_readvariableop_resource:@B
4module_wrapper_dense_biasadd_readvariableop_resource:@I
7module_wrapper_1_dense_1_matmul_readvariableop_resource:@F
8module_wrapper_1_dense_1_biasadd_readvariableop_resource:
identity¢+module_wrapper/dense/BiasAdd/ReadVariableOp¢*module_wrapper/dense/MatMul/ReadVariableOp¢/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp¢.module_wrapper_1/dense_1/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   l
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*module_wrapper/dense/MatMul/ReadVariableOpReadVariableOp3module_wrapper_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¥
module_wrapper/dense/MatMulMatMulflatten/Reshape:output:02module_wrapper/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+module_wrapper/dense/BiasAdd/ReadVariableOpReadVariableOp4module_wrapper_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
module_wrapper/dense/BiasAddBiasAdd%module_wrapper/dense/MatMul:product:03module_wrapper/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
module_wrapper/dense/ReluRelu%module_wrapper/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
.module_wrapper_1/dense_1/MatMul/ReadVariableOpReadVariableOp7module_wrapper_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¼
module_wrapper_1/dense_1/MatMulMatMul'module_wrapper/dense/Relu:activations:06module_wrapper_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/module_wrapper_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp8module_wrapper_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 module_wrapper_1/dense_1/BiasAddBiasAdd)module_wrapper_1/dense_1/MatMul:product:07module_wrapper_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 module_wrapper_1/dense_1/SoftmaxSoftmax)module_wrapper_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
IdentityIdentity*module_wrapper_1/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^module_wrapper/dense/BiasAdd/ReadVariableOp+^module_wrapper/dense/MatMul/ReadVariableOp0^module_wrapper_1/dense_1/BiasAdd/ReadVariableOp/^module_wrapper_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2Z
+module_wrapper/dense/BiasAdd/ReadVariableOp+module_wrapper/dense/BiasAdd/ReadVariableOp2X
*module_wrapper/dense/MatMul/ReadVariableOp*module_wrapper/dense/MatMul/ReadVariableOp2b
/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp2`
.module_wrapper_1/dense_1/MatMul/ReadVariableOp.module_wrapper_1/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Î
+__inference_sequential_layer_call_fn_215990

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_215904o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_215790

args_08
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0y
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameargs_0
³
_
C__inference_flatten_layer_call_and_return_conditional_losses_215760

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

1__inference_module_wrapper_1_layer_call_fn_216114

args_0
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_215831o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameargs_0
Ü

L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_216136

args_08
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0y
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameargs_0
Î

/__inference_module_wrapper_layer_call_fn_216065

args_0
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_215773o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
Ì2
ü	
__inference__traced_save_216223
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_module_wrapper_dense_kernel_read_readvariableop8
4savev2_module_wrapper_dense_bias_read_readvariableop>
:savev2_module_wrapper_1_dense_1_kernel_read_readvariableop<
8savev2_module_wrapper_1_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopA
=savev2_adam_module_wrapper_dense_kernel_m_read_readvariableop?
;savev2_adam_module_wrapper_dense_bias_m_read_readvariableopE
Asavev2_adam_module_wrapper_1_dense_1_kernel_m_read_readvariableopC
?savev2_adam_module_wrapper_1_dense_1_bias_m_read_readvariableopA
=savev2_adam_module_wrapper_dense_kernel_v_read_readvariableop?
;savev2_adam_module_wrapper_dense_bias_v_read_readvariableopE
Asavev2_adam_module_wrapper_1_dense_1_kernel_v_read_readvariableopC
?savev2_adam_module_wrapper_1_dense_1_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*°	
value¦	B£	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_module_wrapper_dense_kernel_read_readvariableop4savev2_module_wrapper_dense_bias_read_readvariableop:savev2_module_wrapper_1_dense_1_kernel_read_readvariableop8savev2_module_wrapper_1_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop=savev2_adam_module_wrapper_dense_kernel_m_read_readvariableop;savev2_adam_module_wrapper_dense_bias_m_read_readvariableopAsavev2_adam_module_wrapper_1_dense_1_kernel_m_read_readvariableop?savev2_adam_module_wrapper_1_dense_1_bias_m_read_readvariableop=savev2_adam_module_wrapper_dense_kernel_v_read_readvariableop;savev2_adam_module_wrapper_dense_bias_v_read_readvariableopAsavev2_adam_module_wrapper_1_dense_1_kernel_v_read_readvariableop?savev2_adam_module_wrapper_1_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesx
v: : : : : : :@:@:@:: : : : :@:@:@::@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 	

_output_shapes
::


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: 
¨

J__inference_module_wrapper_layer_call_and_return_conditional_losses_215773

args_06
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
³
_
C__inference_flatten_layer_call_and_return_conditional_losses_216056

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_216125

args_08
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0y
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameargs_0
¯
Õ
+__inference_sequential_layer_call_fn_215808
flatten_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_215797o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
¨

J__inference_module_wrapper_layer_call_and_return_conditional_losses_216085

args_06
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
¦V
Â
"__inference__traced_restore_216296
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: @
.assignvariableop_5_module_wrapper_dense_kernel:@:
,assignvariableop_6_module_wrapper_dense_bias:@D
2assignvariableop_7_module_wrapper_1_dense_1_kernel:@>
0assignvariableop_8_module_wrapper_1_dense_1_bias:"
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: H
6assignvariableop_13_adam_module_wrapper_dense_kernel_m:@B
4assignvariableop_14_adam_module_wrapper_dense_bias_m:@L
:assignvariableop_15_adam_module_wrapper_1_dense_1_kernel_m:@F
8assignvariableop_16_adam_module_wrapper_1_dense_1_bias_m:H
6assignvariableop_17_adam_module_wrapper_dense_kernel_v:@B
4assignvariableop_18_adam_module_wrapper_dense_bias_v:@L
:assignvariableop_19_adam_module_wrapper_1_dense_1_kernel_v:@F
8assignvariableop_20_adam_module_wrapper_1_dense_1_bias_v:
identity_22¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*°	
value¦	B£	B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp.assignvariableop_5_module_wrapper_dense_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_module_wrapper_dense_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_7AssignVariableOp2assignvariableop_7_module_wrapper_1_dense_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp0assignvariableop_8_module_wrapper_1_dense_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_13AssignVariableOp6assignvariableop_13_adam_module_wrapper_dense_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_14AssignVariableOp4assignvariableop_14_adam_module_wrapper_dense_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_15AssignVariableOp:assignvariableop_15_adam_module_wrapper_1_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_16AssignVariableOp8assignvariableop_16_adam_module_wrapper_1_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_module_wrapper_dense_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_module_wrapper_dense_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_19AssignVariableOp:assignvariableop_19_adam_module_wrapper_1_dense_1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_module_wrapper_1_dense_1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

Ù
F__inference_sequential_layer_call_and_return_conditional_losses_215797

inputs'
module_wrapper_215774:@#
module_wrapper_215776:@)
module_wrapper_1_215791:@%
module_wrapper_1_215793:
identity¢&module_wrapper/StatefulPartitionedCall¢(module_wrapper_1/StatefulPartitionedCall¶
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_215760¢
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0module_wrapper_215774module_wrapper_215776*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_215773¹
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0module_wrapper_1_215791module_wrapper_1_215793*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_215790
IdentityIdentity1module_wrapper_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp'^module_wrapper/StatefulPartitionedCall)^module_wrapper_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Î
+__inference_sequential_layer_call_fn_215977

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_215797o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

à
F__inference_sequential_layer_call_and_return_conditional_losses_215943
flatten_input'
module_wrapper_215932:@#
module_wrapper_215934:@)
module_wrapper_1_215937:@%
module_wrapper_1_215939:
identity¢&module_wrapper/StatefulPartitionedCall¢(module_wrapper_1/StatefulPartitionedCall½
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_215760¢
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0module_wrapper_215932module_wrapper_215934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_215773¹
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0module_wrapper_1_215937module_wrapper_1_215939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_215790
IdentityIdentity1module_wrapper_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp'^module_wrapper/StatefulPartitionedCall)^module_wrapper_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
Î

/__inference_module_wrapper_layer_call_fn_216074

args_0
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_215861o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0

Ù
F__inference_sequential_layer_call_and_return_conditional_losses_215904

inputs'
module_wrapper_215893:@#
module_wrapper_215895:@)
module_wrapper_1_215898:@%
module_wrapper_1_215900:
identity¢&module_wrapper/StatefulPartitionedCall¢(module_wrapper_1/StatefulPartitionedCall¶
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_215760¢
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0module_wrapper_215893module_wrapper_215895*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_215861¹
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0module_wrapper_1_215898module_wrapper_1_215900*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_215831
IdentityIdentity1module_wrapper_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp'^module_wrapper/StatefulPartitionedCall)^module_wrapper_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

J__inference_module_wrapper_layer_call_and_return_conditional_losses_216096

args_06
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
Ò

1__inference_module_wrapper_1_layer_call_fn_216105

args_0
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_215790o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameargs_0

D
(__inference_flatten_layer_call_fn_216050

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_215760`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

à
F__inference_sequential_layer_call_and_return_conditional_losses_215958
flatten_input'
module_wrapper_215947:@#
module_wrapper_215949:@)
module_wrapper_1_215952:@%
module_wrapper_1_215954:
identity¢&module_wrapper/StatefulPartitionedCall¢(module_wrapper_1/StatefulPartitionedCall½
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_215760¢
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0module_wrapper_215947module_wrapper_215949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_215861¹
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0module_wrapper_1_215952module_wrapper_1_215954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_215831
IdentityIdentity1module_wrapper_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp'^module_wrapper/StatefulPartitionedCall)^module_wrapper_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
¨

J__inference_module_wrapper_layer_call_and_return_conditional_losses_215861

args_06
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@
identity¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0u
dense/MatMulMatMulargs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
Ü

L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_215831

args_08
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity¢dense_1/BiasAdd/ReadVariableOp¢dense_1/MatMul/ReadVariableOp
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0y
dense_1/MatMulMatMulargs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameargs_0
×
û
!__inference__wrapped_model_215747
flatten_inputP
>sequential_module_wrapper_dense_matmul_readvariableop_resource:@M
?sequential_module_wrapper_dense_biasadd_readvariableop_resource:@T
Bsequential_module_wrapper_1_dense_1_matmul_readvariableop_resource:@Q
Csequential_module_wrapper_1_dense_1_biasadd_readvariableop_resource:
identity¢6sequential/module_wrapper/dense/BiasAdd/ReadVariableOp¢5sequential/module_wrapper/dense/MatMul/ReadVariableOp¢:sequential/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp¢9sequential/module_wrapper_1/dense_1/MatMul/ReadVariableOpi
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
sequential/flatten/ReshapeReshapeflatten_input!sequential/flatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
5sequential/module_wrapper/dense/MatMul/ReadVariableOpReadVariableOp>sequential_module_wrapper_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Æ
&sequential/module_wrapper/dense/MatMulMatMul#sequential/flatten/Reshape:output:0=sequential/module_wrapper/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@²
6sequential/module_wrapper/dense/BiasAdd/ReadVariableOpReadVariableOp?sequential_module_wrapper_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
'sequential/module_wrapper/dense/BiasAddBiasAdd0sequential/module_wrapper/dense/MatMul:product:0>sequential/module_wrapper/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
$sequential/module_wrapper/dense/ReluRelu0sequential/module_wrapper/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¼
9sequential/module_wrapper_1/dense_1/MatMul/ReadVariableOpReadVariableOpBsequential_module_wrapper_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ý
*sequential/module_wrapper_1/dense_1/MatMulMatMul2sequential/module_wrapper/dense/Relu:activations:0Asequential/module_wrapper_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
:sequential/module_wrapper_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpCsequential_module_wrapper_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0â
+sequential/module_wrapper_1/dense_1/BiasAddBiasAdd4sequential/module_wrapper_1/dense_1/MatMul:product:0Bsequential/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential/module_wrapper_1/dense_1/SoftmaxSoftmax4sequential/module_wrapper_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity5sequential/module_wrapper_1/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
NoOpNoOp7^sequential/module_wrapper/dense/BiasAdd/ReadVariableOp6^sequential/module_wrapper/dense/MatMul/ReadVariableOp;^sequential/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp:^sequential/module_wrapper_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2p
6sequential/module_wrapper/dense/BiasAdd/ReadVariableOp6sequential/module_wrapper/dense/BiasAdd/ReadVariableOp2n
5sequential/module_wrapper/dense/MatMul/ReadVariableOp5sequential/module_wrapper/dense/MatMul/ReadVariableOp2x
:sequential/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp:sequential/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp2v
9sequential/module_wrapper_1/dense_1/MatMul/ReadVariableOp9sequential/module_wrapper_1/dense_1/MatMul/ReadVariableOp:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input

Î
$__inference_signature_wrapper_216045
flatten_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_215747o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
¯
Õ
+__inference_sequential_layer_call_fn_215928
flatten_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_215904o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
Ô
Á
F__inference_sequential_layer_call_and_return_conditional_losses_216010

inputsE
3module_wrapper_dense_matmul_readvariableop_resource:@B
4module_wrapper_dense_biasadd_readvariableop_resource:@I
7module_wrapper_1_dense_1_matmul_readvariableop_resource:@F
8module_wrapper_1_dense_1_biasadd_readvariableop_resource:
identity¢+module_wrapper/dense/BiasAdd/ReadVariableOp¢*module_wrapper/dense/MatMul/ReadVariableOp¢/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp¢.module_wrapper_1/dense_1/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   l
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*module_wrapper/dense/MatMul/ReadVariableOpReadVariableOp3module_wrapper_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¥
module_wrapper/dense/MatMulMatMulflatten/Reshape:output:02module_wrapper/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
+module_wrapper/dense/BiasAdd/ReadVariableOpReadVariableOp4module_wrapper_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0µ
module_wrapper/dense/BiasAddBiasAdd%module_wrapper/dense/MatMul:product:03module_wrapper/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
module_wrapper/dense/ReluRelu%module_wrapper/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@¦
.module_wrapper_1/dense_1/MatMul/ReadVariableOpReadVariableOp7module_wrapper_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0¼
module_wrapper_1/dense_1/MatMulMatMul'module_wrapper/dense/Relu:activations:06module_wrapper_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/module_wrapper_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp8module_wrapper_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 module_wrapper_1/dense_1/BiasAddBiasAdd)module_wrapper_1/dense_1/MatMul:product:07module_wrapper_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 module_wrapper_1/dense_1/SoftmaxSoftmax)module_wrapper_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
IdentityIdentity*module_wrapper_1/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^module_wrapper/dense/BiasAdd/ReadVariableOp+^module_wrapper/dense/MatMul/ReadVariableOp0^module_wrapper_1/dense_1/BiasAdd/ReadVariableOp/^module_wrapper_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2Z
+module_wrapper/dense/BiasAdd/ReadVariableOp+module_wrapper/dense/BiasAdd/ReadVariableOp2X
*module_wrapper/dense/MatMul/ReadVariableOp*module_wrapper/dense/MatMul/ReadVariableOp2b
/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp/module_wrapper_1/dense_1/BiasAdd/ReadVariableOp2`
.module_wrapper_1/dense_1/MatMul/ReadVariableOp.module_wrapper_1/dense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
G
flatten_input6
serving_default_flatten_input:0ÿÿÿÿÿÿÿÿÿD
module_wrapper_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:´h
Á
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
²
_module
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
²
_module
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer

!iter

"beta_1

#beta_2
	$decay
%learning_rate&m`'ma(mb)mc&vd've(vf)vg"
	optimizer
<
&0
'1
(2
)3"
trackable_list_wrapper
<
&0
'1
(2
)3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ú2÷
+__inference_sequential_layer_call_fn_215808
+__inference_sequential_layer_call_fn_215977
+__inference_sequential_layer_call_fn_215990
+__inference_sequential_layer_call_fn_215928À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_sequential_layer_call_and_return_conditional_losses_216010
F__inference_sequential_layer_call_and_return_conditional_losses_216030
F__inference_sequential_layer_call_and_return_conditional_losses_215943
F__inference_sequential_layer_call_and_return_conditional_losses_215958À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÒBÏ
!__inference__wrapped_model_215747flatten_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
/serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_flatten_layer_call_fn_216050¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_216056¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
»

&kernel
'bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
*9&call_and_return_all_conditional_losses
:__call__"
_tf_keras_layer
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
¨2¥
/__inference_module_wrapper_layer_call_fn_216065
/__inference_module_wrapper_layer_call_fn_216074À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Þ2Û
J__inference_module_wrapper_layer_call_and_return_conditional_losses_216085
J__inference_module_wrapper_layer_call_and_return_conditional_losses_216096À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
»

(kernel
)bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
*D&call_and_return_all_conditional_losses
E__call__"
_tf_keras_layer
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
¬2©
1__inference_module_wrapper_1_layer_call_fn_216105
1__inference_module_wrapper_1_layer_call_fn_216114À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
â2ß
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_216125
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_216136À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+@2module_wrapper/dense/kernel
':%@2module_wrapper/dense/bias
1:/@2module_wrapper_1/dense_1/kernel
+:)2module_wrapper_1/dense_1/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÑBÎ
$__inference_signature_wrapper_216045flatten_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
­
5regularization_losses
Mmetrics
Nnon_trainable_variables

Olayers
6	variables
7trainable_variables
Player_regularization_losses
Qlayer_metrics
:__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
­
@regularization_losses
Rmetrics
Snon_trainable_variables

Tlayers
A	variables
Btrainable_variables
Ulayer_regularization_losses
Vlayer_metrics
E__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Wtotal
	Xcount
Y	variables
Z	keras_api"
_tf_keras_metric
^
	[total
	\count
]
_fn_kwargs
^	variables
_	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
W0
X1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
[0
\1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
2:0@2"Adam/module_wrapper/dense/kernel/m
,:*@2 Adam/module_wrapper/dense/bias/m
6:4@2&Adam/module_wrapper_1/dense_1/kernel/m
0:.2$Adam/module_wrapper_1/dense_1/bias/m
2:0@2"Adam/module_wrapper/dense/kernel/v
,:*@2 Adam/module_wrapper/dense/bias/v
6:4@2&Adam/module_wrapper_1/dense_1/kernel/v
0:.2$Adam/module_wrapper_1/dense_1/bias/v©
!__inference__wrapped_model_215747&'()6¢3
,¢)
'$
flatten_inputÿÿÿÿÿÿÿÿÿ
ª "Cª@
>
module_wrapper_1*'
module_wrapper_1ÿÿÿÿÿÿÿÿÿ
C__inference_flatten_layer_call_and_return_conditional_losses_216056X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 w
(__inference_flatten_layer_call_fn_216050K/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¼
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_216125l()?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ@
ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_216136l()?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ@
ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_module_wrapper_1_layer_call_fn_216105_()?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ@
ª

trainingp "ÿÿÿÿÿÿÿÿÿ
1__inference_module_wrapper_1_layer_call_fn_216114_()?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ@
ª

trainingp"ÿÿÿÿÿÿÿÿÿº
J__inference_module_wrapper_layer_call_and_return_conditional_losses_216085l&'?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 º
J__inference_module_wrapper_layer_call_and_return_conditional_losses_216096l&'?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
/__inference_module_wrapper_layer_call_fn_216065_&'?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª

trainingp "ÿÿÿÿÿÿÿÿÿ@
/__inference_module_wrapper_layer_call_fn_216074_&'?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª

trainingp"ÿÿÿÿÿÿÿÿÿ@·
F__inference_sequential_layer_call_and_return_conditional_losses_215943m&'()>¢;
4¢1
'$
flatten_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
F__inference_sequential_layer_call_and_return_conditional_losses_215958m&'()>¢;
4¢1
'$
flatten_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
F__inference_sequential_layer_call_and_return_conditional_losses_216010f&'()7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
F__inference_sequential_layer_call_and_return_conditional_losses_216030f&'()7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_sequential_layer_call_fn_215808`&'()>¢;
4¢1
'$
flatten_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_215928`&'()>¢;
4¢1
'$
flatten_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_215977Y&'()7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_215990Y&'()7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ½
$__inference_signature_wrapper_216045&'()G¢D
¢ 
=ª:
8
flatten_input'$
flatten_inputÿÿÿÿÿÿÿÿÿ"Cª@
>
module_wrapper_1*'
module_wrapper_1ÿÿÿÿÿÿÿÿÿ