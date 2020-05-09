import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AlbertPreTrainedModel, AlbertModel


class AlbertForAnswerSelectionWithConcat(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(
            config.hidden_size, 1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            additional_feature=None,
            labels=None,
            answer_index=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0] # B x L x d
        
        #print(sequence_output.shape)
        
        answer_index = answer_index.unsqueeze(-1).repeat(1,1,sequence_output.shape[-1])

        answer_output = torch.gather(sequence_output,dim = 1, index = answer_index) # B x Nans x d

        #print(answer_output.shape)
        logits = self.classifier(answer_output).squeeze() # B x Nans
        #print(logits.shape)

        #pooled_output = outputs[1]

        #pooled_output = self.dropout(pooled_output)

        #pooled_output = torch.cat((pooled_output, additional_feature), 1)

        #logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]


        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
