# So alias
import tensorflow_decision_forests as tfdf
import collections

# Create the model builder
builder = tfdf.builder.RandomForestBuilder(
    path="/tmp/manual_model",
    objective=tfdf.py_tree.objective.ClassificationObjective(
        label="color", classes=["red", "blue", "green"]))

Tree = tfdf.py_tree.tree.Tree
SimpleColumnSpec = tfdf.py_tree.dataspec.SimpleColumnSpec
ColumnType = tfdf.py_tree.dataspec.ColumnType
# Nodes
NonLeafNode = tfdf.py_tree.node.NonLeafNode
LeafNode = tfdf.py_tree.node.LeafNode
# Conditions
NumericalHigherThanCondition = tfdf.py_tree.condition.NumericalHigherThanCondition
CategoricalIsInCondition = tfdf.py_tree.condition.CategoricalIsInCondition
# Leaf values
ProbabilityValue = tfdf.py_tree.value.ProbabilityValue

builder.add_tree(
    Tree(
        NonLeafNode(
            condition=NumericalHigherThanCondition(
                feature=SimpleColumnSpec(name="f1", type=ColumnType.NUMERICAL),
                threshold=1.5,
                missing_evaluation=False),
            pos_child=NonLeafNode(
                condition=CategoricalIsInCondition(
                    feature=SimpleColumnSpec(name="f2",type=ColumnType.CATEGORICAL),
                    mask=["cat", "dog"],
                    missing_evaluation=False),
                pos_child=LeafNode(value=ProbabilityValue(probability=[0.8, 0.1, 0.1], num_examples=10)),
                neg_child=LeafNode(value=ProbabilityValue(probability=[0.1, 0.8, 0.1], num_examples=20))),
            neg_child=LeafNode(value=ProbabilityValue(probability=[0.1, 0.1, 0.8], num_examples=30)))))

t = Tree(
        NonLeafNode(
            condition=NumericalHigherThanCondition(
                feature=SimpleColumnSpec(name="f1", type=ColumnType.NUMERICAL),
                threshold=1.5,
                missing_evaluation=False),
            pos_child=NonLeafNode(
                condition=CategoricalIsInCondition(
                    feature=SimpleColumnSpec(name="f2",type=ColumnType.CATEGORICAL),
                    mask=["cat", "dog"],
                    missing_evaluation=False),
                pos_child=LeafNode(value=ProbabilityValue(probability=[0.8, 0.1, 0.1], num_examples=10), leaf_idx=2),
                neg_child=LeafNode(value=ProbabilityValue(probability=[0.1, 0.8, 0.1], num_examples=20), leaf_idx=1)),
            neg_child=LeafNode(value=ProbabilityValue(probability=[0.1, 0.1, 0.8], num_examples=30), leaf_idx=0)))


def inOrderTraverse(n):
    if isinstance(n, tfdf.py_tree.node.LeafNode):
        print(n.leaf_idx)
        return None
    inOrderTraverse(n.neg_child)
    # print(n.condition)
    foo = n.condition
    #print(n.leaf_idx.real, n.leaf_idx.imag, n.leaf_idx.numerator, n.leaf_idx.denominator)
    print(foo, foo.missing_evaluation, n.condition.features()[0].name)
    inOrderTraverse(n.pos_child)




print(inOrderTraverse(t.root))




