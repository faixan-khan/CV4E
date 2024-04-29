import random
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    with open(user_submission_file, "r") as test_file_object:
        test_data = json.load(test_file_object)
    
    ground_truth_data = pd.read_csv(test_annotation_file)

    # sorted_json_data = sorted(test_data, key=lambda x: x['image_id'])

    specie_dict = {}
    genus_dict = {}
    family_dict = {}
    for item in test_data:
        specie_dict[item['image_id']] = item['species']
        genus_dict[item['image_id']] = item['Genus']
        family_dict[item['image_id']] = item['family']
    
    image_id = ground_truth_data['image_id'].tolist()
    gt_specie = ground_truth_data['species'].tolist()
    gt_genus = ground_truth_data['Genus'].tolist()
    gt_family = ground_truth_data['family'].tolist()

    specie_list = []
    genus_list = []
    family_list = []
    for i in image_id:
        if i in specie_dict:
            specie_list.append(specie_dict[i])
        else:
            specie_list.append('No prediction')
        if i in genus_dict:
            genus_list.append(genus_dict[i])
        else:
            genus_list.append('No prediction')
        if i in family_dict:
            family_list.append(family_dict[i])
        else:
            family_list.append('No prediction')

#   BEFORE COMMITING TRY THIS ON THE LOCAL MAHICNE TO SEE IT WORKS WITH JSON
    
    # sorted_json_data = sorted(test_data, key=lambda x: x['image_id'])
    # 
    # specie_list = []
    # genus_list = []
    # family_list = []
    # for item in sorted_json_data:
    #     specie_list.append(item['species'])
    #     genus_list.append(item['Genus'])
    #     family_list.append(item['family'])
    # 
    # gt_specie = ground_truth_data['species'].tolist()
    # gt_genus = ground_truth_data['Genus'].tolist()
    # gt_family = ground_truth_data['family'].tolist()

    m_s = confusion_matrix(gt_specie, specie_list)
    m_g = confusion_matrix(gt_genus, genus_list)
    m_f = confusion_matrix(gt_family, family_list)

    spe_acc = np.mean(m_s.diagonal()/ m_s.sum(axis=1))
    gen_acc = np.mean(m_g.diagonal()/ m_g.sum(axis=1))
    fam_acc = np.mean(m_f.diagonal()/ m_f.sum(axis=1))

    print(spe_acc, gen_acc, fam_acc)

    avg_acc = (spe_acc+gen_acc+fam_acc)/3

    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "train_split": {
                    "Family": fam_acc,
                    "Genus": gen_acc,
                    "Specie": spe_acc,
                    "Average": avg_acc,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["train_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "train_split": {
                    "Family": fam_acc,
                    "Genus": gen_acc,
                    "Specie": spe_acc,
                    "Average": avg_acc,
                }
            },
            {
                "test_split": {
                    "Family": fam_acc,
                    "Genus": gen_acc,
                    "Specie": spe_acc,
                    "Average": avg_acc,
                }
            },
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output
