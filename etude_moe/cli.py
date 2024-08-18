import click

import torch.nn as nn

from etude_moe.models import get_classification_accuracy
from etude_moe.data_generation import multiclass_data

from etude_moe.models import Expert
from etude_moe.models import Gate
from etude_moe.models import MoETopX
from etude_moe.models import MoEWeightedAverage
from etude_moe.models import Trainer


@click.group()
def cli() -> None:
    pass


@cli.command(name="test-moe-avg")
def test_moe_avg() -> None:
    # data setup
    num_per_class = 1000
    embedding_size = 64
    num_of_classes = 3
    train_expert_data_x, train_expert_data_y = multiclass_data(
        num_per_class, embedding_size, num_of_classes
    )
    train_moe_data_x, train_moe_data_y = multiclass_data(
        num_per_class, embedding_size, num_of_classes
    )
    test_data_x, test_data_y = multiclass_data(
        num_per_class, embedding_size, num_of_classes
    )

    # train param
    learning_rate = 0.01
    epoches = 3
    batch_size = 4

    # train expert
    num_of_experts = 5
    output_dim = num_of_classes
    experts = [Expert(embedding_size, output_dim) for _ in range(num_of_experts)]
    loss_func = nn.CrossEntropyLoss()
    accuracy_func = get_classification_accuracy
    expert_trainers = [
        Trainer(expert, loss_func, accuracy_func, learning_rate, epoches, batch_size)
        for expert in experts
    ]
    for trainer in expert_trainers:
        trainer.train(train_expert_data_x, train_expert_data_y)

    # train moe
    gate_activate_layer = nn.ReLU()
    gate = Gate(embedding_size, num_of_experts, gate_activate_layer)
    moe = MoEWeightedAverage(experts, gate, embedding_size, output_dim, freeze_expert=True)
    moe_trainer = Trainer(moe, loss_func, accuracy_func, learning_rate, epoches, batch_size)
    moe_trainer.train(train_moe_data_x, train_moe_data_y)

    # evaluate
    for idx, trainer in enumerate(expert_trainers):
        loss, accuracy = trainer.evaluate(test_data_x, test_data_y)
        print(f"expert {idx} - cross entropy {loss}, accuracy: {accuracy}")
    loss, accuracy = moe_trainer.evaluate(test_data_x, test_data_y)
    print(f"moe - cross entropy {loss}, accuracy: {accuracy}")


    moe = MoETopX(
        experts, gate, embedding_size, output_dim, top_x=2, freeze_expert=True,
    )
    moe_trainer = Trainer(moe, loss_func, accuracy_func, learning_rate, epoches, batch_size)
    moe_trainer.train(train_moe_data_x, train_moe_data_y)
    for idx, trainer in enumerate(expert_trainers):
        loss, accuracy = trainer.evaluate(test_data_x, test_data_y)
        print(f"expert {idx} - cross entropy {loss}, accuracy: {accuracy}")
    loss, accuracy = moe_trainer.evaluate(test_data_x, test_data_y)
    print(f"moe - cross entropy {loss}, accuracy: {accuracy}")


if __name__ == "__main__":
    cli()